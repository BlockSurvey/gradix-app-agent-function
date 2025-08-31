"""
Application Evaluation Agent with comprehensive analysis tools
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
# from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI

from config import LLM_MODEL_DEFAULT
from logger.bs_logger import bs_logger

# LangChain tools imports
try:
    from langchain_experimental.tools import PythonREPLTool
    LANGCHAIN_EXPERIMENTAL_AVAILABLE = True
except ImportError:
    LANGCHAIN_EXPERIMENTAL_AVAILABLE = False
    bs_logger.warning("langchain-experimental not available, Python REPL tool disabled")

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class ApplicationEvaluationAgent:
    """Specialized agent for comprehensive application evaluation using multiple tools including LangChain defaults"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL_DEFAULT,
            temperature=0.3
        )
        self._setup_prompt()
        
        # Initialize LangChain default tools
        self._setup_langchain_tools()

    def _setup_prompt(self):
        """Set up the evaluation prompt template"""
        self.system_prompt = """You are an Expert Application Evaluator with access to multiple analysis tools. Your role is to evaluate applications comprehensively using the given grading criteria and rubric.

        Available Tools:
        - Web scraping and analysis
        - GitHub repository analysis
        - Screenshot and visual evaluation
        - Python code execution for calculations and analysis
        - File system operations for data processing
        - Bash shell for system operations
        - JSON data processing for structured analysis
        - Web search for context and validation

        Rules to follow:
        - Understand the given grading criteria and rubric clearly.
        - Understand the given dataset details and metadata clearly.
        - Understand the given application data clearly.
        - Use appropriate tools to gather comprehensive evaluation data.
        - Evaluate the application based on the given grading criteria with all performance levels.
        - Do a detailed evaluation of the application for given criterion level one by one.
        - Grade the application after the evaluation between 1 to 5.
        - Provide detailed feedback with evidence from tool analysis.
        - Use Python calculations for quantitative assessments when needed.
        - Process data files and structures systematically.

        OUTPUT FORMAT: You must return your response in the following JSON format:
        {
            "grade_score": 1, // Numerical score for the given criterion level between 1 to 5
            "grade_feedback": "", // Detailed feedback why the application is graded with the given score
            "tool_evidence": {}, // Evidence gathered from various tools
            "quantitative_analysis": {}, // Results from Python calculations if applicable
            "comprehensive_assessment": "" // Overall comprehensive assessment
        }"""

    async def _setup_langchain_tools(self):
        """Set up LangChain default tools"""
        # search = GoogleSearchAPIWrapper()

        # google_search_tool = Tool(
        #     name="google_search",
        #     description="Search Google for recent results.",
        #     func=search.run,
        # )

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

        # For async environments (e.g., standard Python scripts)
        async_browser = create_async_playwright_browser()
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
        playwright_tools = toolkit.get_tools()

        # github_analyzer tool
        github = GitHubAPIWrapper()
        toolkit = GitHubToolkit.from_github_api_wrapper(github)
        github_tools = toolkit.get_tools()

        # python_repl tool
        python_repl = PythonREPL()
        repl_tool = Tool(
            name="python_repl",
            description="A Python shell to execute commands. Use print(...) to see output.",
            func=python_repl.run,
        )

        self.tools = [
            # google_search_tool, 
            firecrawl_tool,
            playwright_tools,
            github_tools,
            repl_tool
        ]

    async def evaluate_application(
        self,
        application_data: Dict[str, Any],
        grading_info: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive application evaluation using all available tools including LangChain defaults
        
        Args:
            application_data: The application submission data
            grading_info: Grading criteria and rubric
            dataset_info: Dataset context and requirements
            
        Returns:
            Dict containing comprehensive evaluation results
        """
        try:
            bs_logger.info("Starting comprehensive application evaluation with enhanced tools")
            
            evaluation_results = {
                "application_id": application_data.get("id", "unknown"),
                "evaluation_timestamp": datetime.now().isoformat(),
                "langchain_tools_enabled": True,
                "results": {},
                "quantitative_analysis": {},
                "tool_evidence": {}
            }
            
            # Generate AI-powered evaluation with tool evidence
            ai_evaluation = await self._generate_ai_evaluation(
                application_data, grading_info, dataset_info
            )
            evaluation_results["ai_evaluation"] = ai_evaluation
            
            bs_logger.info("Completed comprehensive application evaluation with enhanced tools")
            return evaluation_results
            
        except Exception as e:
            bs_logger.error("Error in enhanced application evaluation: %s", str(e))
            return {
                "error": str(e),
                "application_id": application_data.get("id", "unknown"),
                "evaluation_timestamp": datetime.now().isoformat()
            }

    async def _generate_ai_evaluation(
        self,
        application_data: Dict[str, Any],
        grading_info: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI-powered evaluation based on all collected data including LangChain tool results"""
        try:
            # Prepare structured tool evidence
            # Example tool_results could contain entries like:
            # {"name": "bing_search", "result": "something found..."}
            tool_evidence = "\n".join(
                f"{tool['name'].upper()} RESULT:\n{json.dumps(tool['result'], indent=2)}"
                for tool in self.tools
            )

            context = (
                f"APPLICATION DATA:\n{json.dumps(application_data, indent=2)}\n\n"
                f"GRADING CRITERIA:\n{json.dumps(grading_info, indent=2)}\n\n"
                f"DATASET CONTEXT:\n{json.dumps(dataset_info, indent=2)}\n\n"
                f"TOOL EVIDENCE:\n{tool_evidence}\n"
            )

            human_prompt = (
                "You are evaluating an application for grading purposes. Your task is to provide a "
                "comprehensive, fair, and objective assessment that will determine the applicant's score.\n\n"
                
                "EVALUATION INSTRUCTIONS:\n"
                "1. Carefully review all provided information including application data, grading criteria, "
                "and supporting evidence from various analysis tools.\n"
                "2. Evaluate the application against EACH criterion specified in the grading rubric.\n"
                "3. Base your assessment on concrete evidence and data, not assumptions.\n"
                "4. Be objective and consistent in your scoring methodology.\n\n"
                
                "GRADING REQUIREMENTS:\n"
                "• OVERALL SCORE: Assign a score from 1-5 based on the grading criteria where:\n"
                "  - 5 = Exceptional (Exceeds all requirements)\n"
                "  - 4 = Strong (Meets all requirements with high quality)\n"
                "  - 3 = Satisfactory (Meets core requirements)\n"
                "  - 2 = Below Average (Partially meets requirements)\n"
                "  - 1 = Poor (Fails to meet requirements)\n\n"
                
                "• CRITERION-BY-CRITERION ASSESSMENT:\n"
                "  For each grading criterion, provide:\n"
                "  - Individual score (1-5)\n"
                "  - Specific evidence supporting the score\n"
                "  - How the application meets or fails to meet that criterion\n\n"
                
                "• STRENGTHS ANALYSIS:\n"
                "  - List 3-5 major strengths with specific examples\n"
                "  - Reference tool evidence (GitHub stats, web search results, etc.)\n"
                "  - Highlight exceptional or innovative aspects\n\n"
                
                "• WEAKNESSES & GAPS:\n"
                "  - Identify 3-5 areas needing improvement\n"
                "  - Specify missing requirements or quality issues\n"
                "  - Note any red flags or concerns\n\n"
                
                "• TECHNICAL EVALUATION (if applicable):\n"
                "  - Code quality assessment from GitHub analysis\n"
                "  - Technical architecture and implementation quality\n"
                "  - Documentation and best practices adherence\n\n"
                
                "• DATA & QUANTITATIVE ANALYSIS:\n"
                "  - Key metrics and their significance\n"
                "  - Statistical insights that impact the grade\n"
                "  - Comparison to benchmarks or standards\n\n"
                
                "• FINAL GRADE JUSTIFICATION:\n"
                "  - Clear rationale for the assigned score\n"
                "  - How the score aligns with grading criteria weights\n"
                "  - Overall impression and recommendation\n\n"
                
                "• IMPROVEMENT RECOMMENDATIONS:\n"
                "  - Specific, actionable suggestions for enhancement\n"
                "  - Priority areas to address\n"
                "  - Resources or next steps\n\n"
                
                f"APPLICATION CONTEXT AND DATA:\n\n{context}\n\n"
                
                "Provide your evaluation in a clear, structured format with section headers. "
                "Be thorough but concise, and ensure your grade is well-justified by the evidence presented."
            )

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_prompt),
            ]

            response = self.llm.invoke(messages)

            return {
                "evaluation_text": response.content.strip(),
                "model_used": LLM_MODEL_DEFAULT,
                "tools_utilized": [tool["name"] for tool in self.tools],
                "generated_at": datetime.now().isoformat(),
                "enhanced_with_langchain_tools": True,
            }
        except Exception as e:
            bs_logger.error("Error generating enhanced AI evaluation: %s", str(e))
            return {
                "error": str(e),
                "evaluation_text": (
                    "Enhanced AI evaluation could not be generated due to technical issues."
                )
            }
