from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from database.firebase_db import firestore_client
from controllers.user_controller import UserController
from controllers.weather_controller import WeatherController, WeatherQuery

app = FastAPI(
    title="Gradix Agent API",
    description="AI-powered agent functions for the Gradix platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "Gradix Agent API",
        "version": "1.0.0"
    }

# User endpoints
@app.get("/users/{user_id}")
def get_user(user_id: int):
    user_controller = UserController()
    return user_controller.get_user(user_id)

# Weather endpoints
@app.post("/weather/sf")
async def get_sf_weather():
    """Get current weather in San Francisco using AI agent"""
    weather_controller = WeatherController()
    return await weather_controller.get_sf_weather_with_agent()

@app.get("/weather/sf/raw")
def get_sf_weather_raw():
    """Get raw weather data for San Francisco"""
    weather_controller = WeatherController()
    return weather_controller.get_sf_weather_raw()

@app.post("/weather/query")
async def query_weather(weather_query: WeatherQuery):
    """Query weather for any city using AI agent"""
    weather_controller = WeatherController()
    return await weather_controller.query_weather(weather_query)