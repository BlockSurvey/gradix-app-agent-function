from fastapi import APIRouter

from .grading import router as grading_router

api_router = APIRouter()

api_router.include_router(grading_router)