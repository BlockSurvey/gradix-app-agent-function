from fastapi import APIRouter

from .grading import router as grading_router
from .weather import router as weather_router

api_router = APIRouter()

api_router.include_router(grading_router)
api_router.include_router(weather_router)