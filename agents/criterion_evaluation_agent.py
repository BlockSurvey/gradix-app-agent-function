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

from bs4 import BeautifulSoup
import requests


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
        
        # # web_crawl_tool
        # def web_crawl_tool(url: str) -> str:
        #     """Crawl a website and extract content using FireCrawl"""
        #     try:
        #         from config import FIRECRAWL_API_KEY
        #         loader = FireCrawlLoader(
        #             api_key=FIRECRAWL_API_KEY,
        #             url=url,
        #             mode="crawl"
        #         )
        #         docs = loader.load()
        #         return "\n".join([doc.page_content for doc in docs])
        #     except Exception as e:
        #         return f"Error crawling website: {str(e)}"

        # firecrawl_tool = Tool(
        #     name="web_crawl",
        #     description="Crawl and extract content from websites using FireCrawl. Input should be a valid URL.",
        #     func=web_crawl_tool,
        # )

        def simple_scraper(url: str) -> str:
            """Scrape website content with URL cleaning"""
            try:
                # Clean the URL - remove quotes, commas, and whitespace
                cleaned_url = url.strip()
                # Remove surrounding quotes if present
                if cleaned_url.startswith('"') and cleaned_url.endswith('"'):
                    cleaned_url = cleaned_url[1:-1]
                elif cleaned_url.startswith("'") and cleaned_url.endswith("'"):
                    cleaned_url = cleaned_url[1:-1]
                # Remove trailing comma if present
                if cleaned_url.endswith(','):
                    cleaned_url = cleaned_url[:-1]
                # Remove any remaining whitespace
                cleaned_url = cleaned_url.strip()
                
                # Validate URL format
                if not cleaned_url.startswith(('http://', 'https://')):
                    return f"Error: Invalid URL format. URL must start with http:// or https://. Got: {cleaned_url}"
                
                # Make the request with timeout
                response = requests.get(cleaned_url, timeout=30)
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove unnecessary tags like nav, footer, script, style
                for tag in soup.select("nav, footer, script, style"):
                    tag.decompose()
                    
                # Get clean text
                text = soup.get_text(separator="\n").strip()
                
                # Limit response size
                if len(text) > 10000:
                    text = text[:10000] + "\n... (truncated)"
                    
                return text
                
            except requests.exceptions.RequestException as e:
                return f"Error fetching URL '{cleaned_url}': {str(e)}"
            except Exception as e:
                return f"Error processing URL '{url}': {str(e)}"

        scraper_tool = Tool(
            name="BeautifulSoup Web Scraper",
            func=simple_scraper,
            description=(
                "Scrapes the given URL and returns clean text content. "
                "The URL should be a valid HTTP/HTTPS URL. "
                "Automatically cleans the URL by removing quotes, commas, and extra whitespace."
            ),
        )

        # GitHub repository analyzer tool for public repos
        def analyze_github_repo(repo_url: str) -> str:
            """Analyze a GitHub repository and extract key information"""
            try:
                import requests
                import re
                
                # Clean the URL
                cleaned_url = repo_url.strip()
                if cleaned_url.startswith('"') and cleaned_url.endswith('"'):
                    cleaned_url = cleaned_url[1:-1]
                elif cleaned_url.startswith("'") and cleaned_url.endswith("'"):
                    cleaned_url = cleaned_url[1:-1]
                cleaned_url = cleaned_url.strip()
                
                # Extract owner and repo from GitHub URL
                github_pattern = r'github\.com/([^/]+)/([^/]+)'
                match = re.search(github_pattern, cleaned_url)
                
                if not match:
                    return f"Error: Invalid GitHub URL format. Expected format: https://github.com/owner/repo"
                
                owner, repo = match.groups()
                # Remove .git suffix if present
                repo = repo.replace('.git', '')
                
                # Use GitHub API to get repository information
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                
                response = requests.get(api_url, timeout=30)
                response.raise_for_status()
                
                repo_data = response.json()
                
                # Extract key information
                info = []
                info.append(f"Repository: {repo_data.get('full_name', 'N/A')}")
                info.append(f"Description: {repo_data.get('description', 'No description')}")
                info.append(f"Language: {repo_data.get('language', 'Not specified')}")
                info.append(f"Stars: {repo_data.get('stargazers_count', 0)}")
                info.append(f"Forks: {repo_data.get('forks_count', 0)}")
                info.append(f"Open Issues: {repo_data.get('open_issues_count', 0)}")
                info.append(f"Created: {repo_data.get('created_at', 'N/A')}")
                info.append(f"Last Updated: {repo_data.get('updated_at', 'N/A')}")
                info.append(f"Size: {repo_data.get('size', 0)} KB")
                
                # Get README content if available
                try:
                    readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
                    readme_response = requests.get(readme_url, timeout=30)
                    if readme_response.status_code == 200:
                        readme_data = readme_response.json()
                        import base64
                        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                        # Limit README content
                        if len(readme_content) > 2000:
                            readme_content = readme_content[:2000] + "\n... (truncated)"
                        info.append(f"\nREADME Content:\n{readme_content}")
                except:
                    info.append("\nREADME: Not available or could not be fetched")
                
                return "\n".join(info)
                
            except requests.exceptions.RequestException as e:
                return f"Error fetching GitHub repository '{repo_url}': {str(e)}"
            except Exception as e:
                return f"Error analyzing GitHub repository '{repo_url}': {str(e)}"

        github_analyzer_tool = Tool(
            name="github_analyzer",
            description="Analyze a GitHub repository and extract key information including description, language, stats, and README content. Provide the full GitHub repository URL.",
            func=analyze_github_repo,
        )
        
        github_tools = [github_analyzer_tool]

        # python_repl tool
        python_repl = PythonREPL()
        repl_tool = Tool(
            name="python_repl",
            description="A Python shell to execute commands. Use print(...) to see output. Can be used for calculations, data analysis, and file processing.",
            func=python_repl.run,
        )

        # Data analysis tool with input cleaning
        def analyze_json_data(json_str: str) -> str:
            """Analyze JSON data structure and provide insights"""
            try:
                import json
                
                # Clean the input - remove surrounding quotes if present
                cleaned_input = json_str.strip()
                if cleaned_input.startswith('"') and cleaned_input.endswith('"'):
                    cleaned_input = cleaned_input[1:-1]
                elif cleaned_input.startswith("'") and cleaned_input.endswith("'"):
                    cleaned_input = cleaned_input[1:-1]
                
                # Try to parse as JSON
                data = json.loads(cleaned_input)
                
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
            except json.JSONDecodeError as e:
                return f"Error parsing JSON: {str(e)}. Input: {json_str[:100]}..."
            except Exception as e:
                return f"Error analyzing JSON: {str(e)}"
        
        json_analyzer_tool = Tool(
            name="json_analyzer",
            description="Analyze JSON data structure to understand its content and organization. Automatically cleans input by removing surrounding quotes.",
            func=analyze_json_data,
        )

        self.tools = [
            scraper_tool,
            github_tools,
            json_analyzer_tool,
            repl_tool,
        ]

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
            
            # Create memory for the agent with proper output key configuration
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"  # Explicitly set output key to avoid warning
            )
            
            # Prepare all context as a single formatted string
            formatted_context = f"""
SYSTEM CONTEXT:
{self.system_prompt}

APPLICATION DATA:
{json.dumps(application_data, indent=2)}

DATASET CONTEXT:
{json.dumps(dataset_context, indent=2) if dataset_context else "No dataset context provided"}

CRITERION TO EVALUATE: {self.criterion_name}
"""
            
            # Create the ReAct agent prompt with single input variable
            prompt_template = """You are evaluating an application for a specific criterion. Here is the full context:

{input}

You have access to the following tools:

{tools}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
1. Always follow the exact format above. Each line must start with either "Thought:", "Action:", "Action Input:", "Observation:", or "Final Answer:".
2. Your Final Answer MUST be a valid JSON object with this structure:
{{
    "grade_score": <number between 1-5>,
    "grade_evidence": [<list of specific evidence strings>],
    "grade_feedback": "<detailed feedback explaining the grade>"
}}

Previous conversation:
{chat_history}

Begin! Remember to always use the exact format specified above.

{agent_scratchpad}"""

            agent_prompt = PromptTemplate(
                input_variables=["input", "tools", "tool_names", "agent_scratchpad", "chat_history"],
                template=prompt_template
            )
            
            # Create the ReAct agent
            agent = create_react_agent(
                llm=self.llm,
                tools=flat_tools,
                prompt=agent_prompt
            )
            
            # Create the agent executor with better error handling
            agent_executor = AgentExecutor(
                agent=agent,
                tools=flat_tools,
                memory=memory,
                verbose=True,
                max_iterations=10,  # Reduced to prevent infinite loops
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                max_execution_time=60  # Add timeout
            )
            
            # Prepare single input string with all context and instructions
            evaluation_request = f"""{formatted_context}

TASK: Please evaluate the application for the '{self.criterion_name}' criterion.

INSTRUCTIONS:
1. Use the available tools to gather evidence about the application
2. Analyze how well the application meets the criterion requirements  
3. Assign a score from 1-5 based on the criterion levels:
   - 5 = Exceptional (Exceeds all requirements)
   - 4 = Strong (Meets all requirements with high quality)
   - 3 = Satisfactory (Meets core requirements)
   - 2 = Below Average (Partially meets requirements)
   - 1 = Poor (Fails to meet requirements)
4. Provide specific evidence supporting your score
5. Write detailed feedback explaining your evaluation

Remember: Your Final Answer must be a valid JSON object."""

            # Execute the agent with single input
            result = await agent_executor.ainvoke({"input": evaluation_request})
            
            # Extract the evaluation from the result
            evaluation_text = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Try to parse JSON from the evaluation text
            evaluation_data = self._parse_evaluation_json(evaluation_text)
            
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
                "grade_score": evaluation_data.get("grade_score", self._extract_score(evaluation_text)),
                "grade_evidence": evaluation_data.get("grade_evidence", []),
                "grade_feedback": evaluation_data.get("grade_feedback", evaluation_text),
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
            
            APPLICATION CONTEXT:
            {context}
            
            REQUIRED OUTPUT FORMAT:
            You must provide your evaluation as a JSON object with this exact structure:
            {{
                "grade_score": <number between 1-5>,
                "grade_evidence": [<list of specific evidence strings from the application>],
                "grade_feedback": "<detailed feedback explaining why this grade was assigned>"
            }}
            
            EVALUATION GUIDELINES:
            1. SCORE (1-5): Assign a numerical score based on the criterion levels where:
               - 5 = Exceptional (Exceeds all requirements)
               - 4 = Strong (Meets all requirements with high quality)
               - 3 = Satisfactory (Meets core requirements)
               - 2 = Below Average (Partially meets requirements)
               - 1 = Poor (Fails to meet requirements)
            
            2. EVIDENCE: List 3-5 specific pieces of evidence from the application that support your score
            
            3. FEEDBACK: Provide detailed feedback that includes:
               - What the application does well regarding {self.criterion_name}
               - What aspects need improvement
               - Specific recommendations for enhancement
            
            Remember: Focus ONLY on {self.criterion_name}. Do not evaluate other aspects.
            
            Provide your evaluation as a valid JSON object:
            """
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Try to parse the JSON response
            evaluation_data = self._parse_evaluation_json(response_text)
            
            # If JSON parsing failed, create structured data from text
            if not evaluation_data or 'grade_score' not in evaluation_data:
                score = self._extract_score(response_text)
                evaluation_data = {
                    "grade_score": score,
                    "grade_evidence": [],
                    "grade_feedback": response_text
                }
            
            return {
                "criterion": self.criterion_name,
                "weight": self.criterion_details.get('weight', 1.0),
                "evaluation": response_text,
                "grade_score": evaluation_data.get("grade_score", 3),
                "grade_evidence": evaluation_data.get("grade_evidence", []),
                "grade_feedback": evaluation_data.get("grade_feedback", response_text),
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

    def _parse_evaluation_json(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse JSON from evaluation text"""
        import re
        import json
        
        try:
            # Try to find JSON in the text
            json_patterns = [
                r'\{[^{}]*"grade_score"[^{}]*\}',  # Simple JSON object
                r'\{.*?"grade_score".*?\}(?=\s*$)',  # JSON at end
                r'```json\s*(.*?)\s*```',  # JSON in code block
                r'```\s*(.*?)\s*```',  # JSON in generic code block
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, evaluation_text, re.DOTALL | re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            # Clean up the match
                            json_str = match.strip()
                            if json_str.startswith('```'):
                                json_str = json_str[3:]
                            if json_str.endswith('```'):
                                json_str = json_str[:-3]
                            if json_str.startswith('json'):
                                json_str = json_str[4:]
                            
                            # Parse the JSON
                            data = json.loads(json_str)
                            if isinstance(data, dict) and 'grade_score' in data:
                                return data
                        except json.JSONDecodeError:
                            continue
            
            # If no JSON found, try to extract directly from text
            return {}
            
        except Exception as e:
            bs_logger.warning(f"Could not parse JSON from evaluation: {str(e)}")
            return {}


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