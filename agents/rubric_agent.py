import json
import uuid
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from logger.bs_logger import bs_logger
from config import LLM_MODEL_DEFAULT


class RubricAgent:
    """Agent for generating comprehensive grading rubrics"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL_DEFAULT,
            temperature=0.1
        )
        self._setup_prompt()

    def _setup_prompt(self):
        """Set up the prompt template for rubric generation"""
        self.system_prompt = """You are an Expert Application Grader specializing in creating comprehensive grading rubrics for the given Grading Type and Evaluation Criteria.
        Your role is to understand the given Grading Type and Evaluation Criteria clearly and
        Create a clear criteria with 5 performance levels that can be applied against the given application, project, submission, etc.

        Rules to follow:
            - The criteria must be specific to the given Grading Type and Evaluation Criteria.
            - The criteria must be clear and concise.
            - The criteria must be measurable.
            - The criteria must be achievable.

        Remember: The criteria must be defined from the given Grading Type and Evaluation Criteria. You must consider the criteria from the given Evaluation Criteria and not add any additional criteria.

        OUTPUT FORMAT: You must return your response in the following JSON format:
        {
            "criteriaDetails": [ // List of criteria
                {
                    "id": "", // uuid
                    "name": "", // Name of the criterion
                    "weight": "", // It will be out of 100
                    "description": "", // Detailed description of the criterion to evaluate against the given application, project, submission, etc.
                    "levels": [
                        {
                            "level": 1,
                            "description": "" // Clear description of the performance level
                        },
                        {
                            "level": 2,
                            "description": "" // Clear description of the performance level
                        },
                        {
                            "level": 3,
                            "description": "" // Clear description of the performance level
                        },
                        {
                            "level": 4,
                            "description": "" // Clear description of the performance level
                        },
                        {
                            "level": 5,
                            "description": "" // Clear description of the performance level
                        }
                    ]
                }
            ]
        }"""

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
            Create a grading rubric for:

            **Rubric Name**: {name}
            **Grading Type**: {grading_type}
            **Evaluation Criteria**: {criteria}

            Requirements:
            - Use ONLY the provided evaluation criteria
            - Create 5 performance levels (1=Inadequate, 2=Developing, 3=Satisfactory, 4=Proficient, 5=Exemplary)
            - Break down criteria into individual components if multiple aspects are mentioned
            - Assign weights that total 100 across all criteria
            - Generate a unique UUID for each criterion

            Return the response in the JSON format specified in the system prompt.
            """

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_prompt)
            ]

            bs_logger.info("Generating rubric for: %s (%s)", name, grading_type)

            response = self.llm.invoke(messages)

            # Parse the JSON response with improved handling
            try:
                content = response.content.strip()
                
                # Handle ```json format
                if content.startswith('```json'):
                    # Extract JSON from code block
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        content = content[json_start:json_end]
                
                rubric_data = json.loads(content)

                # Add metadata
                rubric_data["generated_by"] = "RubricAgent"
                rubric_data["generation_model"] = LLM_MODEL_DEFAULT
                rubric_data["rubric_name"] = name
                rubric_data["grading_type"] = grading_type
                rubric_data["original_criteria"] = criteria

                bs_logger.info("Successfully generated rubric for: %s", name)
                return rubric_data

            except (json.JSONDecodeError, ValueError) as e:
                bs_logger.warning("JSON parsing failed: %s, using fallback", str(e))
                return self._create_fallback_rubric(name, grading_type, criteria, response.content)

        except Exception as e:
            bs_logger.error("Error generating rubric: %s", e)
            raise

    def _create_fallback_rubric(self, name: str, grading_type: str, criteria: str, raw_response: str) -> Dict[str, Any]:
        """Create a fallback rubric structure if JSON parsing fails"""

        return {
            "criteriaDetails": [
                {
                    "id": str(uuid.uuid4()),
                    "name": f"Overall {criteria.split(',')[0].strip() if ',' in criteria else criteria[:50]}",
                    "weight": "100",
                    "levels": [
                        {
                            "level": 1,
                            "description": "Inadequate - Fails to meet basic requirements and expectations for the specified criteria"
                        },
                        {
                            "level": 2,
                            "description": "Developing - Partially meets requirements with noticeable deficiencies in the specified criteria"
                        },
                        {
                            "level": 3,
                            "description": "Satisfactory - Meets basic requirements and expectations adequately for the specified criteria"
                        },
                        {
                            "level": 4,
                            "description": "Proficient - Meets all requirements with high quality execution of the specified criteria"
                        },
                        {
                            "level": 5,
                            "description": "Exemplary - Exceeds all expectations and requirements, demonstrating exceptional quality in the specified criteria"
                        }
                    ]
                }
            ],
            "generated_by": "RubricAgent",
            "generation_model": LLM_MODEL_DEFAULT,
            "rubric_name": name,
            "grading_type": grading_type,
            "original_criteria": criteria,
            "fallback_used": True,
            "raw_response": raw_response
        }
