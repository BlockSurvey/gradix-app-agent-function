from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, Any
import json

from logger.bs_logger import bs_logger
from config import LLM_MODEL_DEFAULT


class RubricAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL_DEFAULT,
            temperature=0.1
        )
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Set up the prompt template for rubric generation"""
        self.system_prompt = """You are an expert educational assessment designer. Your task is to create detailed, comprehensive rubric grading criteria with 5 performance levels.

For any grading task, you must create:

1. **Performance Levels** (5 levels from highest to lowest):
   - Level 5: Exemplary/Outstanding
   - Level 4: Proficient/Good
   - Level 3: Satisfactory/Average
   - Level 2: Developing/Below Average
   - Level 1: Inadequate/Poor

2. **Clear Definitions**: Each level must have:
   - Specific, measurable criteria
   - Observable behaviors or outcomes
   - Clear distinction from other levels
   - Actionable feedback indicators

3. **Comprehensive Coverage**: Address all aspects of the grading criteria provided

4. **Consistency**: Maintain consistent language and structure across all levels

Return the rubric as a structured JSON object with clear, detailed descriptions for each performance level."""

    def create_detailed_rubric(self, name: str, grading_type: str, criteria: str) -> Dict[str, Any]:
        """
        Generate a detailed 5-level rubric based on the provided grading details
        
        Args:
            name: Name/title of the grading rubric
            grading_type: Type of grading (e.g., essay, project, presentation)
            criteria: Specific criteria to be evaluated
            
        Returns:
            Dict containing the detailed rubric structure
        """
        try:
            human_prompt = f"""
            Create a detailed grading rubric with the following specifications:
            
            **Rubric Name**: {name}
            **Grading Type**: {grading_type}
            **Evaluation Criteria**: {criteria}
            
            Generate a comprehensive 5-level performance rubric that includes:
            
            1. Clear performance level definitions (1-5 scale)
            2. Specific criteria for each level
            3. Observable behaviors/outcomes for each level
            4. Point ranges or percentages for each level
            5. Detailed feedback indicators
            
            Structure the response as a JSON object with the following format:
            {{
                "rubric_name": "{name}",
                "grading_type": "{grading_type}",
                "total_points": 100,
                "performance_levels": {{
                    "level_5": {{
                        "name": "Exemplary",
                        "point_range": "90-100",
                        "description": "Detailed description...",
                        "criteria": ["specific criterion 1", "specific criterion 2", ...]
                    }},
                    "level_4": {{
                        "name": "Proficient",
                        "point_range": "80-89",
                        "description": "Detailed description...",
                        "criteria": ["specific criterion 1", "specific criterion 2", ...]
                    }},
                    "level_3": {{
                        "name": "Satisfactory",
                        "point_range": "70-79",
                        "description": "Detailed description...",
                        "criteria": ["specific criterion 1", "specific criterion 2", ...]
                    }},
                    "level_2": {{
                        "name": "Developing",
                        "point_range": "60-69",
                        "description": "Detailed description...",
                        "criteria": ["specific criterion 1", "specific criterion 2", ...]
                    }},
                    "level_1": {{
                        "name": "Inadequate",
                        "point_range": "0-59",
                        "description": "Detailed description...",
                        "criteria": ["specific criterion 1", "specific criterion 2", ...]
                    }}
                }},
                "usage_guidelines": ["guideline 1", "guideline 2", ...],
                "feedback_template": "Template for providing feedback based on performance level"
            }}
            
            Ensure each level has distinct, measurable criteria that clearly differentiate performance quality.
            """
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            bs_logger.info(f"Generating rubric for: {name} ({grading_type})")
            
            response = self.llm.invoke(messages)
            
            # Parse the JSON response
            try:
                rubric_data = json.loads(response.content)
                
                # Add metadata
                rubric_data["generated_by"] = "RubricAgent"
                rubric_data["generation_model"] = LLM_MODEL_DEFAULT
                
                bs_logger.info(f"Successfully generated rubric for: {name}")
                return rubric_data
                
            except json.JSONDecodeError:
                # Fallback: return structured response even if JSON parsing fails
                bs_logger.warning("JSON parsing failed, returning structured fallback")
                return self._create_fallback_rubric(name, grading_type, criteria, response.content)
                
        except Exception as e:
            bs_logger.error(f"Error generating rubric: {e}")
            raise
    
    def _create_fallback_rubric(self, name: str, grading_type: str, criteria: str, raw_response: str) -> Dict[str, Any]:
        """Create a fallback rubric structure if JSON parsing fails"""
        return {
            "rubric_name": name,
            "grading_type": grading_type,
            "total_points": 100,
            "performance_levels": {
                "level_5": {
                    "name": "Exemplary",
                    "point_range": "90-100",
                    "description": "Outstanding performance exceeding expectations",
                    "criteria": ["Exceeds all requirements", "Shows exceptional understanding"]
                },
                "level_4": {
                    "name": "Proficient",
                    "point_range": "80-89",
                    "description": "Good performance meeting most expectations",
                    "criteria": ["Meets most requirements", "Shows good understanding"]
                },
                "level_3": {
                    "name": "Satisfactory",
                    "point_range": "70-79",
                    "description": "Adequate performance meeting basic expectations",
                    "criteria": ["Meets basic requirements", "Shows adequate understanding"]
                },
                "level_2": {
                    "name": "Developing",
                    "point_range": "60-69",
                    "description": "Below average performance with room for improvement",
                    "criteria": ["Meets some requirements", "Shows limited understanding"]
                },
                "level_1": {
                    "name": "Inadequate",
                    "point_range": "0-59",
                    "description": "Poor performance not meeting basic expectations",
                    "criteria": ["Fails to meet requirements", "Shows little understanding"]
                }
            },
            "usage_guidelines": [
                "Use this rubric consistently for all assessments",
                "Provide specific feedback based on performance level",
                "Consider multiple aspects of the work when assigning scores"
            ],
            "feedback_template": "Based on the rubric criteria, the work demonstrates [performance level] with specific strengths in [areas] and opportunities for improvement in [areas]",
            "generated_by": "RubricAgent",
            "generation_model": LLM_MODEL_DEFAULT,
            "fallback_used": True,
            "raw_response": raw_response
        }