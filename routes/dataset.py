from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Dict, Any
from controllers.dataset_controller import DatasetController

router = APIRouter(prefix="/datasets", tags=["datasets"])
dataset_controller = DatasetController()


@router.post("/upload", response_model=Dict[str, Any])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset file and store dataset metadata and applications in Firebase.
    
    - **file**: CSV file containing application data
    
    Returns:
    - Dataset ID and metadata including columns, number of applications, etc.
    """
    return await dataset_controller.upload_dataset(file)


@router.get("/", response_model=List[Dict[str, Any]])
async def get_datasets(
    limit: int = Query(50, ge=1, le=100, description="Number of datasets to retrieve"),
    offset: int = Query(0, ge=0, description="Number of datasets to skip for pagination")
):
    """
    Get a list of all datasets with pagination.
    
    - **limit**: Maximum number of datasets to return (1-100)
    - **offset**: Number of datasets to skip for pagination
    
    Returns:
    - List of dataset metadata
    """
    return await dataset_controller.get_datasets(limit=limit, offset=offset)


@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset(dataset_id: str):
    """
    Get dataset metadata by ID.
    
    - **dataset_id**: Unique identifier for the dataset
    
    Returns:
    - Dataset metadata including file details, columns, number of applications
    """
    return await dataset_controller.get_dataset(dataset_id)


@router.get("/{dataset_id}/applications", response_model=List[Dict[str, Any]])
async def get_dataset_applications(
    dataset_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Number of applications to retrieve"),
    offset: int = Query(0, ge=0, description="Number of applications to skip for pagination")
):
    """
    Get applications from a specific dataset with pagination.
    
    - **dataset_id**: Unique identifier for the dataset
    - **limit**: Maximum number of applications to return (1-1000)
    - **offset**: Number of applications to skip for pagination
    
    Returns:
    - List of applications (each row from the CSV as JSON)
    """
    return await dataset_controller.get_applications(dataset_id, limit=limit, offset=offset)


@router.delete("/{dataset_id}", response_model=Dict[str, str])
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset and all its applications.
    
    - **dataset_id**: Unique identifier for the dataset
    
    Returns:
    - Confirmation message
    """
    return await dataset_controller.delete_dataset(dataset_id)


@router.get("/{dataset_id}/summary", response_model=Dict[str, Any])
async def get_ai_dataset_summary(dataset_id: str):
    """
    Generate or retrieve AI-powered dataset summary.
    
    This endpoint:
    - Generates a comprehensive paragraph-style summary using AI
    - Saves the summary to Firebase for future use (caching)
    - Returns cached summary if already generated
    
    The summary includes insights about:
    - Dataset purpose and classification
    - Machine learning opportunities
    - Business value and ROI potential
    
    - **dataset_id**: Unique identifier for the dataset
    
    Returns:
    - dataset_id: The dataset identifier
    - summary: Paragraph-style AI-generated summary
    - summary_generated_at: Timestamp when summary was created
    - cached: Boolean indicating if this is a cached response
    - message: Status message (only for new summaries)
    """
    return await dataset_controller.get_ai_dataset_summary(dataset_id)
