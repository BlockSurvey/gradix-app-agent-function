from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from controllers.grading_controller import GradingController

router = APIRouter(
    prefix="/gradings",
    tags=["grading"],
    responses={404: {"description": "Not found"}},
)

class GradingRequest(BaseModel):
    name: str
    type: str
    criteria: str

class GradingUpdate(BaseModel):
    name: str = None
    type: str = None
    criteria: str = None

@router.post("/")
def create_grading(grading_request: GradingRequest):
    """Create a new grading entry"""
    grading_controller = GradingController()
    grading_data = {
        "name": grading_request.name,
        "type": grading_request.type,
        "criteria": grading_request.criteria
    }
    return grading_controller.create_grading(grading_data)

@router.get("/{grading_id}")
def get_grading(grading_id: str):
    """Get grading by ID"""
    grading_controller = GradingController()
    return grading_controller.get_grading(grading_id)

# @router.put("/{grading_id}")
# def update_grading(grading_id: str, grading_update: GradingUpdate):
#     """Update grading by ID"""
#     grading_controller = GradingController()
#     update_data = {k: v for k, v in grading_update.dict().items() if v is not None}
#     return grading_controller.update_grading(grading_id, update_data)

# @router.delete("/{grading_id}")
# def delete_grading(grading_id: str):
#     """Delete grading by ID"""
#     grading_controller = GradingController()
#     return grading_controller.delete_grading(grading_id)

@router.post("/{grading_id}/rubric")
def generate_rubric(grading_id: str):
    """Generate detailed rubric for a grading"""
    grading_controller = GradingController()
    return grading_controller.generate_rubric(grading_id)

@router.get("/")
def list_gradings(limit: int = 10, offset: int = 0):
    """List all gradings with pagination"""
    grading_controller = GradingController()
    return grading_controller.list_gradings(limit)

@router.get("/{grading_id}/status")
def get_grading_status(grading_id: str):
    """Get grading status and progress"""
    grading_controller = GradingController()
    grading = grading_controller.get_grading(grading_id)
    if not grading:
        raise HTTPException(status_code=404, detail="Grading not found")
    return {
        "grading_id": grading_id,
        "status": grading.get("status", "unknown"),
        "progress": grading.get("progress", 0)
    }