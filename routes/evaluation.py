"""
Evaluation API routes for the Gradix platform
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from controllers.dataset_controller import EvaluationController
from services.evaluation_service import EvaluationService
from logger.bs_logger import bs_logger

router = APIRouter()

# Initialize controllers and services
evaluation_controller = EvaluationController()
evaluation_service = EvaluationService()


# Pydantic models for request/response validation
class CreateEvaluationRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    grading_id: str
    dataset_id: str
    settings: Optional[Dict[str, Any]] = {}


class EvaluationResponse(BaseModel):
    success: bool
    evaluation_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class EvaluationListResponse(BaseModel):
    success: bool
    evaluations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


@router.post("/evaluations", response_model=EvaluationResponse)
async def create_evaluation(request: CreateEvaluationRequest):
    """
    Create a new evaluation linking grading criteria with a dataset
    """
    try:
        bs_logger.info(f"Creating evaluation: {request.name}")
        
        result = await evaluation_controller.create_evaluation(request.model_dump())
        
        if result["success"]:
            bs_logger.info(f"Evaluation created successfully: {result['evaluation_id']}")
        else:
            bs_logger.error(f"Failed to create evaluation: {result.get('error')}")
        
        return EvaluationResponse(**result)
        
    except Exception as e:
        bs_logger.error(f"Error creating evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
def get_evaluation(evaluation_id: str):
    """
    Get evaluation details by ID
    """
    try:
        result = evaluation_controller.get_evaluation(evaluation_id)
        
        if result["success"]:
            return EvaluationResponse(**result)
        else:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])
        
    except HTTPException:
        raise
    except Exception as e:
        bs_logger.error(f"Error getting evaluation {evaluation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations", response_model=EvaluationListResponse)
def list_evaluations(limit: int = 10, status: Optional[str] = None):
    """
    List evaluations with optional status filter
    """
    try:
        result = evaluation_controller.list_evaluations(limit=limit, status=status)
        return EvaluationListResponse(**result)
        
    except Exception as e:
        bs_logger.error(f"Error listing evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluations/{evaluation_id}/start")
async def start_evaluation(evaluation_id: str):
    """
    Start processing the first application in an evaluation
    """
    try:
        bs_logger.info(f"Starting evaluation: {evaluation_id}")
        
        result = await evaluation_controller.start_evaluation(evaluation_id)
        
        if result["success"]:
            bs_logger.info(f"Evaluation {evaluation_id} started successfully")
            return result
        else:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])
        
    except HTTPException:
        raise
    except Exception as e:
        bs_logger.error(f"Error starting evaluation {evaluation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
