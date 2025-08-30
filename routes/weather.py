from fastapi import APIRouter
from controllers.weather_controller import WeatherController, WeatherQuery

router = APIRouter(
    prefix="/weather",
    tags=["weather"],
    responses={404: {"description": "Not found"}},
)

@router.post("/sf")
async def get_sf_weather():
    """Get current weather in San Francisco using AI agent"""
    weather_controller = WeatherController()
    return await weather_controller.get_sf_weather_with_agent()

@router.get("/sf/raw")
def get_sf_weather_raw():
    """Get raw weather data for San Francisco"""
    weather_controller = WeatherController()
    return weather_controller.get_sf_weather_raw()

@router.post("/query")
async def query_weather(weather_query: WeatherQuery):
    """Query weather for any city using AI agent"""
    weather_controller = WeatherController()
    return await weather_controller.query_weather(weather_query)

@router.get("/cities")
def get_supported_cities():
    """Get list of supported cities for weather queries"""
    return {
        "supported_cities": [
            "San Francisco", "New York", "London", "Paris", "Tokyo", 
            "Sydney", "Mumbai", "Berlin", "Toronto", "Los Angeles"
        ],
        "note": "Weather data can be fetched for any city worldwide"
    }

@router.get("/forecast/{city}")
async def get_city_forecast(city: str, days: int = 3):
    """Get weather forecast for a specific city"""
    weather_controller = WeatherController()
    weather_query = WeatherQuery(query=f"Get {days} day forecast for {city}")
    return await weather_controller.query_weather(weather_query)