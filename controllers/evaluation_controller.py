"""Evaluation controller for handling evaluation operations."""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from agents.application_evaluation_agent import ApplicationEvaluationAgent
# Import the multi-agent orchestrator
from agents.criterion_evaluation_agent import CriterionEvaluationAgent, MultiAgentEvaluationOrchestrator
from controllers.dataset_controller import DatasetController
from controllers.grading_controller import GradingController
from database.firebase_db import firestore_client
from logger.bs_logger import bs_logger


class EvaluationController:
    """Controller for handling evaluation operations that combine grading and dataset processing."""

    def __init__(self):
        self.db = firestore_client
        self.grading_controller = GradingController()
        self.dataset_controller = DatasetController()
        self.evaluations_collection = "evaluations"
        self.results_collection = "evaluation_results"

    async def create_evaluation(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new evaluation that links grading criteria with a dataset.

        Args:
            evaluation_data: Dictionary containing grading_id, dataset_id, and evaluation settings

        Returns:
            Dictionary with created evaluation ID and data
        """
        try:
            # Validate required fields
            required_fields = ["grading_id", "dataset_id", "name"]
            for field in required_fields:
                if field not in evaluation_data:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }

            # Verify grading exists
            grading_result = self.grading_controller.get_grading(evaluation_data["grading_id"])
            if not grading_result["success"]:
                return {
                    "success": False,
                    "error": "Invalid grading_id: " + grading_result.get("error", "Grading not found")
                }

            # Verify dataset exists
            try:
                dataset_info = await self.dataset_controller.get_dataset(
                    evaluation_data["dataset_id"]
                )
            except Exception as dataset_error:
                return {
                    "success": False,
                    "error": f"Invalid dataset_id: {str(dataset_error)}"
                }

            # Generate unique evaluation ID
            evaluation_id = str(uuid.uuid4())

            # Prepare evaluation document
            evaluation_doc = {
                "id": evaluation_id,
                "name": evaluation_data["name"],
                "description": evaluation_data.get("description", ""),
                "grading_id": evaluation_data["grading_id"],
                "dataset_id": evaluation_data["dataset_id"],
                "status": "created",
                "settings": evaluation_data.get("settings", {}),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "grading_info": grading_result["data"],
                "dataset_info": {
                    "filename": dataset_info.get("filename"),
                    "num_applications": dataset_info.get("num_applications"),
                    "columns": dataset_info.get("columns"),
                    "ai_summary": dataset_info.get("ai_summary")
                }
            }

            # Store evaluation in Firebase
            self.db.collection(self.evaluations_collection).document(evaluation_id).set(evaluation_doc)

            bs_logger.info("Evaluation %s created successfully", evaluation_id)

            return {
                "success": True,
                "evaluation_id": evaluation_id,
                "data": evaluation_doc
            }

        except Exception as error:
            bs_logger.error("Error creating evaluation: %s", str(error))
            return {
                "success": False,
                "error": str(error)
            }

    async def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation by ID."""
        try:
            doc = self.db.collection(self.evaluations_collection).document(evaluation_id).get()
            if doc.exists:
                return {
                    "success": True,
                    "data": doc.to_dict(),
                    "evaluation_id": evaluation_id
                }
            return {
                "success": False,
                "error": "Evaluation not found"
            }
        except Exception as error:
            return {
                "success": False,
                "error": str(error)
            }

    async def list_evaluations(self, limit: int = 10, status: str = None) -> Dict[str, Any]:
        """List evaluations with optional status filter."""
        try:
            query = self.db.collection(self.evaluations_collection)

            if status:
                query = query.where("status", "==", status)

            docs = query.order_by("created_at", direction="DESCENDING").limit(limit).stream()
            evaluations = []

            for doc in docs:
                evaluation_data = doc.to_dict()
                evaluation_data["evaluation_id"] = doc.id
                evaluations.append(evaluation_data)

            return {
                "success": True,
                "evaluations": evaluations
            }
        except Exception as error:
            return {
                "success": False,
                "error": str(error)
            }

    async def start_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Start processing applications through the evaluation."""
        try:
            # Get evaluation details
            evaluation_result = await self.get_evaluation(evaluation_id)
            if not evaluation_result["success"]:
                return evaluation_result

            evaluation_data = evaluation_result["data"]

            # Update evaluation status to processing
            self.db.collection(self.evaluations_collection).document(evaluation_id).update({
                "status": "processing",
                "started_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            })

            # Get the first application from the dataset for processing
            dataset_id = evaluation_data["dataset_id"]
            applications = await self.dataset_controller.get_applications(dataset_id, limit=1)
            if applications:
                first_app = applications[0]

                # Process the first application using the comprehensive evaluation agent
                result = await self.process_single_application(
                    evaluation_id,
                    first_app["data"],
                    evaluation_data["grading_info"],
                    evaluation_data["dataset_info"]
                )

                return {
                    "success": True,
                    "evaluation_id": evaluation_id,
                    "status": "processing",
                    "total_applications": len(applications),
                    "first_application_result": result
                }

            return {
                "success": False,
                "error": "No applications found in dataset"
            }

        except Exception as error:
            bs_logger.error("Error starting evaluation %s: %s", evaluation_id, str(error))
            return {
                "success": False,
                "error": str(error)
            }

    def _score_application_criterion(self, app_data: Dict, criterion_name: str) -> float:
        """Score a single criterion for an application."""
        # Simple keyword-based scoring
        criterion_keywords = criterion_name.lower().split()
        app_text = " ".join([str(v).lower() for v in app_data.values() if v is not None])

        # Count keyword matches
        matches = sum(1 for keyword in criterion_keywords if keyword in app_text)

        # Return score based on matches (0-5 scale)
        if matches >= len(criterion_keywords):
            return 5.0  # Excellent
        if matches >= len(criterion_keywords) * 0.7:
            return 4.0  # Good
        if matches >= len(criterion_keywords) * 0.5:
            return 3.0  # Satisfactory
        if matches >= len(criterion_keywords) * 0.3:
            return 2.0  # Needs improvement
        return 1.0  # Inadequate

    def _basic_application_score(self, app_data: Dict, criteria: str) -> float:
        """Basic scoring when no detailed rubric is available."""
        # Count non-empty fields
        non_empty_fields = sum(1 for v in app_data.values()
                             if v is not None and str(v).strip())
        total_fields = len(app_data)

        if total_fields == 0:
            return 1.0

        # Completeness score
        completeness = non_empty_fields / total_fields

        # Keywords from criteria
        if criteria:
            criteria_keywords = criteria.lower().split()
            app_text = " ".join([str(v).lower() for v in app_data.values() if v is not None])
            keyword_matches = sum(1 for keyword in criteria_keywords if keyword in app_text)
            keyword_score = min(keyword_matches / max(len(criteria_keywords), 1), 1.0)
        else:
            keyword_score = 0.5  # Neutral if no criteria

        # Combined score (weighted)
        return (completeness * 0.6 + keyword_score * 0.4) * 5  # Scale to 1-5

    def _calculate_grade(self, total_score: float) -> str:
        """Convert numeric score to letter grade."""
        if total_score >= 4.5:
            return "A"
        if total_score >= 3.5:
            return "B"
        if total_score >= 2.5:
            return "C"
        if total_score >= 1.5:
            return "D"
        return "F"

    def _generate_feedback(self, total_score: float) -> str:
        """Generate feedback based on scores."""
        if total_score >= 4.0:
            return "Excellent application that meets or exceeds expectations."
        if total_score >= 3.0:
            return "Good application with strong qualifications."
        if total_score >= 2.0:
            return "Satisfactory application with room for improvement."
        return "Application needs significant improvement to meet requirements."

    async def _store_evaluation_results(self, results: Dict[str, Any]) -> None:
        try:
            result_id = str(uuid.uuid4())
            results_doc = {
                "id": result_id,
                **results,
                "stored_at": datetime.now(timezone.utc)
            }
            
            self.db.collection(self.results_collection).document(result_id).set(results_doc)

            print(f"Evaluation results stored successfully: ------------------")
        except Exception as e:
            bs_logger.error(
                "Failed to store evaluation results for application %s: %s",
                results.get("application_id"),
                str(e)
            )

    async def process_single_application(
        self,
        evaluation_id: str,
        application_data: Dict[str, Any],
        grading_info: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            criterion_evaluation_results = []

            # Let us evaluate all criteria one by one
            rubric = grading_info.get('rubric', {})
            criteria_details = rubric.get('criteriaDetails', [])
            for criterion in criteria_details:
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
                agent_result = await agent.evaluate_criterion(
                    application_data=application_data,
                    dataset_context=dataset_info
                )
                criterion_evaluation_results.append(agent_result)
            
            # Store the criterion evaluation results
            await self._store_evaluation_results({
                "criterion_evaluation_results": criterion_evaluation_results
            })

            return {
                "success": True,
                "criterion_evaluation_results": criterion_evaluation_results
            }

        except Exception as error:
            bs_logger.error("Error in multi-agent processing for application: %s", str(error))
            return {
                "success": False,
                "error": str(error),
                "evaluation_id": evaluation_id,
                "application_id": application_data.get("id", "unknown"),
                "evaluation_method": "multi_agent_architecture"
            }