"""
Criterion-specific evaluation agent for multi-agent architecture
"""
import base64
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_community.tools.playwright.utils import \
    create_async_playwright_browser
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.document_loaders import FireCrawlLoader
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

from config import LLM_MODEL_DEFAULT, OPENAI_API_KEY
from logger.bs_logger import bs_logger


class CriterionEvaluationAgent:
    """Agent responsible for evaluating a single criterion"""
    
    def __init__(self, criterion_name: str, criterion_details: Dict[str, Any]):
        """
        Initialize the criterion evaluation agent
        
        Args:
            criterion_name: Name of the criterion to evaluate
            criterion_details: Details about the criterion including levels, weights, description
        """
        self.criterion_name = criterion_name
        self.criterion_details = criterion_details
        self.llm = ChatOpenAI(
            model=LLM_MODEL_DEFAULT,
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        self.system_prompt = self._setup_prompt()
    
    def _setup_prompt(self) -> str:
        """Setup the system prompt for this specific criterion"""
        levels_description = self._format_levels()
        
        return f"""
        You are an Expert Application Evaluator with access to multiple analysis tools.
        You are a specialized evaluation agent focused exclusively on assessing the '{self.criterion_name}' criterion.
        
        Your role is to provide a detailed, objective evaluation of ONLY this specific criterion.
        
        CRITERION DETAILS:
        Name: {self.criterion_name}
        Description: {self.criterion_details.get('description', 'No description provided')}
        Weight: {self.criterion_details.get('weight', 'Not specified')}
        
        SCORING LEVELS:
        {levels_description}

        EVALUATION GUIDELINES:
        - Understand the given grading criteria and rubric clearly.
        - Focus only on the given criterion.
        - Understand the given dataset details and metadata clearly.
        - Understand the given application data clearly.
        - Use appropriate tools to gather comprehensive evaluation data.
        - Evaluate the application based on the given grading criteria with all performance levels.
        - Do a detailed evaluation of the application for given criterion level one by one.
        - Grade the application after the evaluation between 1 to 5.
        - Provide detailed feedback with evidence from tool analysis.
        - Use Python calculations for quantitative assessments when needed.
        - Process data files and structures systematically.
        
        IMPORTANT: Perform a extensive evaluation of the given application by given criterion and levels. You can use the tools to evaluation the given application by given criterion and levels."""
    
    async def _setup_langchain_tools(self):
        """Set up LangChain default tools"""
        # search = GoogleSearchAPIWrapper()

        # google_search_tool = Tool(
        #     name="google_search",
        #     description="Search Google for recent results.",
        #     func=search.run,
        # )

        # web_search_tool
        search_tool = DuckDuckGoSearchRun()
        web_search_tool = Tool(
            name="web_search",
            description="Search the web for information. Input should be a search query string.",
            func=search_tool.run,
        )

        # web_crawl_tool
        def web_crawl_tool(url: str) -> str:
            """Crawl a website and extract content using FireCrawl"""
            try:
                from config import FIRECRAWL_API_KEY
                loader = FireCrawlLoader(
                    api_key=FIRECRAWL_API_KEY,
                    url=url,
                    mode="crawl"
                )
                docs = loader.load()
                return "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                return f"Error crawling website: {str(e)}"

        firecrawl_tool = Tool(
            name="web_crawl",
            description="Crawl and extract content from websites using FireCrawl. Input should be a valid URL.",
            func=web_crawl_tool,
        )

        # Skip Playwright tools in async environments to avoid event loop conflicts
        # Instead, create simple web scraping tool using aiohttp
        async def async_web_fetch(url: str) -> str:
            """Fetch web content asynchronously"""
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            text = soup.get_text()
                            # Clean up text
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)
                            return text[:5000]  # Limit to first 5000 chars
                        else:
                            return f"Error: HTTP {response.status}"
            except Exception as e:
                return f"Error fetching URL: {str(e)}"
        
        # Create synchronous wrapper for async web fetch
        import asyncio
        def web_fetch_sync(url: str) -> str:
            """Synchronous wrapper for async web fetch"""
            try:
                # Check if there's already a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, async_web_fetch(url))
                        return future.result(timeout=30)
                except RuntimeError:
                    # No running loop, we can use asyncio.run
                    return asyncio.run(async_web_fetch(url))
            except Exception as e:
                return f"Error in web fetch: {str(e)}"

        web_fetch_tool = Tool(
            name="web_fetch",
            description="Fetch and extract text content from a webpage. Input should be a valid URL.",
            func=web_fetch_sync,
        )

        # github_analyzer tool - only if credentials are available
        try:
            github = GitHubAPIWrapper()
            toolkit = GitHubToolkit.from_github_api_wrapper(github)
            github_tools = toolkit.get_tools()
        except Exception as e:
            bs_logger.warning(f"GitHub tools not available: {str(e)}")
            github_tools = []

        # python_repl tool
        python_repl = PythonREPL()
        repl_tool = Tool(
            name="python_repl",
            description="A Python shell to execute commands. Use print(...) to see output. Can be used for calculations, data analysis, and file processing.",
            func=python_repl.run,
        )

        # Data analysis tool
        def analyze_json_data(json_str: str) -> str:
            """Analyze JSON data structure and provide insights"""
            try:
                import json
                data = json.loads(json_str)
                
                analysis = []
                analysis.append(f"Data type: {type(data).__name__}")
                
                if isinstance(data, dict):
                    analysis.append(f"Number of keys: {len(data)}")
                    analysis.append(f"Keys: {', '.join(list(data.keys())[:10])}")
                elif isinstance(data, list):
                    analysis.append(f"Number of items: {len(data)}")
                    if data and isinstance(data[0], dict):
                        analysis.append(f"First item keys: {', '.join(data[0].keys())}")
                
                return "\n".join(analysis)
            except Exception as e:
                return f"Error analyzing JSON: {str(e)}"
        
        json_analyzer_tool = Tool(
            name="json_analyzer",
            description="Analyze JSON data structure to understand its content and organization.",
            func=analyze_json_data,
        )

        self.tools = [
            # google_search_tool, 
            web_search_tool,
            firecrawl_tool,
            web_fetch_tool,
            json_analyzer_tool,
            repl_tool
        ]
        
        # Add GitHub tools if available
        if github_tools:
            self.tools.extend(github_tools)

    
    def _format_levels(self) -> str:
        """Format the criterion levels for the prompt"""
        levels = self.criterion_details.get('levels', [])
        if not levels:
            return "No specific levels defined - use standard 1-5 scale"
        
        formatted = []
        for level_info in levels:
            level_num = level_info.get('level', 'Unknown')
            description = level_info.get('description', 'No description')
            formatted.append(f"Level {level_num}: {description}")
        
        return "\n".join(formatted)
    
    async def evaluate_criterion(
        self,
        application_data: Dict[str, Any],
        dataset_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the application for this specific criterion using LangChain agent with tools
        
        Args:
            application_data: The application data to evaluate
            dataset_context: Optional dataset context
            
        Returns:
            Evaluation results for this criterion
        """
        try:
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain.prompts import PromptTemplate
            from langchain.memory import ConversationBufferMemory
            
            # Setup tools for the agent
            await self._setup_langchain_tools()
            
            # Flatten nested tools lists
            flat_tools = []
            for tool in self.tools:
                if isinstance(tool, list):
                    flat_tools.extend(tool)
                else:
                    flat_tools.append(tool)
            
            # Create memory for the agent
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create the ReAct agent prompt
            agent_prompt = PromptTemplate.from_template("""
            {system_prompt}
            
            You have access to the following tools:
            {tools}
            
            Use the following format:
            
            Thought: Consider what you need to evaluate for the criterion
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I have gathered enough information to evaluate the criterion
            Final Answer: The complete evaluation with score, justification, evidence, strengths, weaknesses, and recommendations
            
            APPLICATION DATA:
            {application_data}
            
            DATASET CONTEXT:
            {dataset_context}
            
            Chat History:
            {chat_history}
            
            Question: Evaluate this application SPECIFICALLY for the '{criterion_name}' criterion. 
            
            OUTPUT FORMAT: You must return your response in the following JSON format:
            {
                "grade_score": 1, // Numerical score for the given criterion level between 1 to 5
                "grade_evidence": [], // List of evidence for the given criterion level as string array
                "grade_feedback": "", // Detailed feedback why the application is graded with the given score
            }
            
            Begin evaluation:
            {input}
            
            {agent_scratchpad}
            """)
            
            # Create the ReAct agent
            agent = create_react_agent(
                llm=self.llm,
                tools=flat_tools,
                prompt=agent_prompt
            )
            
            # Create the agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=flat_tools,
                memory=memory,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            # Prepare the input for the agent
            agent_input = {
                "system_prompt": self.system_prompt,
                "application_data": json.dumps(application_data, indent=2),
                "dataset_context": json.dumps(dataset_context, indent=2) if dataset_context else "No dataset context provided",
                "criterion_name": self.criterion_name,
                "input": f"Please evaluate the application for the '{self.criterion_name}' criterion. Use the available tools to gather comprehensive evidence and perform detailed analysis."
            }
            
            # Execute the agent
            result = await agent_executor.ainvoke(agent_input)
            
            # Extract the evaluation from the result
            evaluation_text = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Format tool usage evidence
            tool_evidence = []
            for action, observation in intermediate_steps:
                if hasattr(action, 'tool'):
                    tool_evidence.append({
                        "tool": action.tool,
                        "input": str(action.tool_input)[:500],  # Truncate long inputs
                        "summary": str(observation)[:500] if observation else "No output"
                    })
            
            return {
                "criterion": self.criterion_name,
                "weight": self.criterion_details.get('weight', 1.0),
                "evaluation": evaluation_text,
                "score": self._extract_score(evaluation_text),
                "tool_evidence": tool_evidence,
                "evaluated_at": datetime.now().isoformat(),
                "agent_type": "criterion_specific_with_tools",
                "model_used": LLM_MODEL_DEFAULT,
                "tools_used": [tool.name if hasattr(tool, 'name') else str(tool) for tool in flat_tools]
            }
            
        except Exception as e:
            bs_logger.error(f"Error evaluating criterion {self.criterion_name}: {str(e)}")
            
            # Fallback to simpler evaluation without tools if agent fails
            try:
                return await self._fallback_evaluation(application_data, dataset_context)
            except Exception as fallback_error:
                bs_logger.error(f"Fallback evaluation also failed: {str(fallback_error)}")
                return {
                    "criterion": self.criterion_name,
                    "error": str(e),
                    "fallback_error": str(fallback_error),
                    "score": 0,
                    "evaluation": f"Failed to evaluate {self.criterion_name} due to technical error"
                }

    
    async def _fallback_evaluation(
        self,
        application_data: Dict[str, Any],
        dataset_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Fallback evaluation method without tools if the agent-based evaluation fails
        
        Args:
            application_data: The application data to evaluate
            dataset_context: Optional dataset context
            
        Returns:
            Evaluation results for this criterion
        """
        try:
            context_parts = [
                f"APPLICATION DATA:\n{json.dumps(application_data, indent=2)}"
            ]
            
            if dataset_context:
                context_parts.append(
                    f"DATASET CONTEXT:\n{json.dumps(dataset_context, indent=2)}"
                )
            
            context = "\n\n".join(context_parts)
            
            human_prompt = f"""
            Evaluate this application SPECIFICALLY for the '{self.criterion_name}' criterion.
            
            REQUIRED OUTPUT FORMAT:
            
            1. SCORE (1-5): Assign a numerical score based on the defined levels
            
            2. LEVEL JUSTIFICATION:
               - Which level does this application achieve?
               - What specific requirements does it meet or fail to meet?
            
            3. EVIDENCE:
               - List 3-5 specific pieces of evidence from the application
               - Quote or reference exact data points
            
            4. STRENGTHS (for this criterion only):
               - What does the application do well regarding {self.criterion_name}?
               - Specific examples of excellence
            
            5. WEAKNESSES (for this criterion only):
               - What aspects of {self.criterion_name} need improvement?
               - Specific gaps or issues
            
            6. RECOMMENDATIONS:
               - How can the applicant improve on {self.criterion_name}?
               - Specific, actionable suggestions
            
            APPLICATION CONTEXT:
            {context}
            
            Remember: Focus ONLY on {self.criterion_name}. Do not evaluate other aspects.
            """
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            return {
                "criterion": self.criterion_name,
                "weight": self.criterion_details.get('weight', 1.0),
                "evaluation": response.content.strip(),
                "score": self._extract_score(response.content),
                "evaluated_at": datetime.now().isoformat(),
                "agent_type": "criterion_specific_fallback",
                "model_used": LLM_MODEL_DEFAULT,
                "fallback_reason": "Agent with tools failed, using direct LLM evaluation"
            }
            
        except Exception as e:
            bs_logger.error(f"Fallback evaluation failed for {self.criterion_name}: {str(e)}")
            raise
    
    def _extract_score(self, evaluation_text: str) -> float:
        """Extract the numerical score from the evaluation text"""
        import re

        # Look for patterns like "SCORE: 4" or "Score (1-5): 3"
        patterns = [
            r'SCORE[:\s]*(\d+)',
            r'Score[:\s]*(\d+)',
            r'score[:\s]*(\d+)',
            r'(\d+)\s*/\s*5',
            r'Level[:\s]*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 1 <= score <= 5:
                        return score
                except:
                    continue
        
        # Default to 3 if no score found
        bs_logger.warning(f"Could not extract score for {self.criterion_name}, defaulting to 3")
        return 3.0


class MultiAgentEvaluationOrchestrator:
    """Orchestrates multiple criterion agents for comprehensive evaluation"""
    
    def __init__(self):
        """Initialize the orchestrator"""
        self.criterion_agents = []
        self.merger_llm = ChatOpenAI(
            model=LLM_MODEL_DEFAULT,
            temperature=0.2,
            api_key=OPENAI_API_KEY
        )
    
    def setup_criterion_agents(self, grading_criteria: Dict[str, Any]) -> None:
        """
        Setup individual agents for each criterion
        
        Args:
            grading_criteria: Dictionary containing all criteria and their details
        """
        self.criterion_agents: List[CriterionEvaluationAgent] = []

        rubric = grading_criteria.get('rubric', {})
        for criterion in rubric.get('criteriaDetails', []):
            name = criterion.get('name', '')
            description = criterion.get('description', '')
            weight = criterion.get('weight', '')
            levels = criterion.get('levels', [])
            # Get the criterion details from the rubric.criteriaDetails
            criterion_details = {
                'name': name,
                'description': description,
                'weight': weight,
                'levels': levels
            }

            agent = CriterionEvaluationAgent(name, criterion_details)
            self.criterion_agents.append(agent)
            bs_logger.info(f"Created agent for criterion: {name}")
    
    async def evaluate_all_criteria(
        self,
        application_data: Dict[str, Any],
        grading_info: Dict[str, Any],
        dataset_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate application using all criterion agents
        
        Args:
            application_data: Application to evaluate
            grading_info: Grading rubric information
            dataset_info: Optional dataset context
            
        Returns:
            Combined evaluation results
        """
        try:
            # Setup agents based on grading criteria
            self.setup_criterion_agents(grading_info)
            
            if not self.criterion_agents:
                raise ValueError("No criterion agents configured")
            
            # Evaluate each criterion in parallel (simulated with sequential for now)
            criterion_evaluations = []
            
            for agent in self.criterion_agents:
                bs_logger.info(f"Evaluating criterion: {agent.criterion_name}")
                
                evaluation = await agent.evaluate_criterion(
                    application_data=application_data,
                    dataset_context=dataset_info,
                )
                
                criterion_evaluations.append(evaluation)
                bs_logger.info(f"Completed evaluation for: {agent.criterion_name}, Score: {evaluation.get('score', 'N/A')}")
            
            # Merge all evaluations
            merged_results = await self._merge_evaluations(
                criterion_evaluations,
                application_data,
                grading_info
            )
            
            return {
                "success": True,
                "individual_evaluations": criterion_evaluations,
                "merged_evaluation": merged_results,
                "total_criteria_evaluated": len(criterion_evaluations),
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_method": "multi_agent_architecture"
            }
            
        except Exception as e:
            bs_logger.error(f"Error in multi-agent evaluation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "evaluation_method": "multi_agent_architecture"
            }
    
    async def _merge_evaluations(
        self,
        criterion_evaluations: List[Dict[str, Any]],
        application_data: Dict[str, Any],
        grading_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge individual criterion evaluations into a comprehensive result
        
        Args:
            criterion_evaluations: List of individual criterion evaluations
            application_data: Original application data
            grading_info: Grading rubric information
            
        Returns:
            Merged evaluation results
        """
        try:
            # Calculate weighted average score
            total_weight = 0
            weighted_sum = 0
            
            for eval in criterion_evaluations:
                if 'error' not in eval:
                    # Convert weight to float, handling string inputs
                    weight = eval.get('weight', 1.0)
                    if isinstance(weight, str):
                        try:
                            # Remove any percentage signs and convert
                            weight_str = weight.replace('%', '').strip()
                            weight = float(weight_str)
                            # If it was a percentage, convert to decimal
                            if '%' in eval.get('weight', ''):
                                weight = weight / 100.0
                        except (ValueError, AttributeError):
                            bs_logger.warning(f"Could not parse weight '{weight}' for criterion {eval.get('criterion', 'Unknown')}, using 1.0")
                            weight = 1.0
                    else:
                        weight = float(weight) if weight else 1.0
                    
                    score = float(eval.get('score', 0))
                    weighted_sum += score * weight
                    total_weight += weight
            
            final_score = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Prepare summary for LLM merger
            evaluations_summary = "\n\n".join([
                f"CRITERION: {eval['criterion']}\n"
                f"SCORE: {eval.get('score', 'N/A')}\n"
                f"WEIGHT: {eval.get('weight', 1.0)}\n"
                f"EVALUATION:\n{eval.get('evaluation', 'No evaluation provided')}"
                for eval in criterion_evaluations
            ])
            
            # Use LLM to create coherent merged evaluation
            merger_prompt = f"""
            You are tasked with merging multiple criterion-specific evaluations into a comprehensive final evaluation.
            
            INDIVIDUAL CRITERION EVALUATIONS:
            {evaluations_summary}
            
            CALCULATED WEIGHTED SCORE: {final_score:.2f} / 5.0
            
            APPLICATION INFO:
            Applicant: {application_data.get('name', 'Unknown')}
            Application ID: {application_data.get('id', 'Unknown')}
            
            TASK:
            1. Synthesize all criterion evaluations into a coherent overall assessment
            2. Confirm or adjust the final score based on the holistic view
            3. Identify the top 3 strengths across all criteria
            4. Identify the top 3 areas for improvement across all criteria
            5. Provide an overall recommendation
            
            FORMAT YOUR RESPONSE AS:
            
            FINAL SCORE: [Confirm {final_score:.2f} or suggest adjustment with justification]
            
            OVERALL ASSESSMENT:
            [2-3 paragraph synthesis of all evaluations]
            
            KEY STRENGTHS:
            1. [Strength 1]
            2. [Strength 2]
            3. [Strength 3]
            
            AREAS FOR IMPROVEMENT:
            1. [Area 1]
            2. [Area 2]
            3. [Area 3]
            
            CRITERION BREAKDOWN:
            [Brief summary of how each criterion was scored]
            
            FINAL RECOMMENDATION:
            [Clear recommendation based on all evaluations]
            """
            
            messages = [
                SystemMessage(content="You are an expert evaluation synthesizer responsible for creating coherent final assessments from multiple criterion evaluations."),
                HumanMessage(content=merger_prompt)
            ]
            
            response = await self.merger_llm.ainvoke(messages)
            
            return {
                "final_score": final_score,
                "merged_evaluation": response.content.strip(),
                "criterion_scores": {
                    eval['criterion']: {
                        "score": eval.get('score', 0),
                        "weight": eval.get('weight', 1.0)
                    }
                    for eval in criterion_evaluations
                },
                "total_criteria": len(criterion_evaluations),
                "evaluation_method": "weighted_average_with_llm_synthesis",
                "merged_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            bs_logger.error(f"Error merging evaluations: {str(e)}")
            return {
                "error": str(e),
                "final_score": 0,
                "merged_evaluation": "Failed to merge evaluations due to technical error"
            }