from fastapi import APIRouter

from .grading import router as grading_router
from .dataset import router as dataset_router
from .evaluation import router as evaluation_router

api_router = APIRouter()

api_router.include_router(grading_router)
api_router.include_router(dataset_router)
api_router.include_router(evaluation_router)