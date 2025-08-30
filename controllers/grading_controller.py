"""Grading controller for managing grading data and rubric generation."""
from datetime import datetime, timezone
from typing import Dict
from database.firebase_db import firestore_client
from agents.rubric_agent import RubricAgent


class GradingController:
    """Controller for handling grading operations and rubric generation."""
    def __init__(self):
        self.db = firestore_client
        self.collection_name = "gradings"
        self.rubric_agent = RubricAgent()

    def create_grading(self, grading_data: Dict) -> Dict:
        """
        Create and store grading details in Firebase
        
        Args:
            grading_data: Dictionary containing name, type, and criteria
            
        Returns:
            Dictionary with created grading ID and data
        """
        try:
            # Add timestamp
            grading_data['created_at'] = datetime.now(timezone.utc)
            grading_data['updated_at'] = datetime.now(timezone.utc)
            # Add to Firebase
            doc_ref = self.db.collection(self.collection_name).add(grading_data)
            grading_id = doc_ref[1].id
            
            return {
                "success": True,
                "grading_id": grading_id,
                "data": grading_data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_grading(self, grading_id: str) -> Dict:
        """Get grading by ID"""
        try:
            doc = self.db.collection(self.collection_name).document(grading_id).get()
            if doc.exists:
                return {
                    "success": True,
                    "data": doc.to_dict(),
                    "grading_id": grading_id
                }
            else:
                return {
                    "success": False,
                    "error": "Grading not found"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_rubric(self, grading_id: str) -> Dict:
        """
        Generate detailed rubric with 5 performance levels for a grading
        
        Args:
            grading_id: ID of the grading to generate rubric for
            
        Returns:
            Dictionary with generated rubric
        """
        try:
            # Get grading details
            grading_result = self.get_grading(grading_id)
            if not grading_result["success"]:
                return grading_result
            
            grading_data = grading_result["data"]
            
            # Generate rubric using the agent
            rubric = self.rubric_agent.create_detailed_rubric(
                name=grading_data.get("name", ""),
                grading_type=grading_data.get("type", ""),
                criteria=grading_data.get("criteria", "")
            )
            
            # Store the generated rubric back to the grading document
            self.db.collection(self.collection_name).document(grading_id).update({
                "rubric": rubric,
                "rubric_generated_at": datetime.now(timezone.utc)
            })
            
            return {
                "success": True,
                "grading_id": grading_id,
                "rubric": rubric
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_gradings(self, limit: int = 10) -> Dict:
        """List all gradings with pagination"""
        try:
            docs = self.db.collection(self.collection_name).limit(limit).stream()
            gradings = []
            
            for doc in docs:
                grading_data = doc.to_dict()
                grading_data["grading_id"] = doc.id
                gradings.append(grading_data)
            
            return {
                "success": True,
                "gradings": gradings
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }