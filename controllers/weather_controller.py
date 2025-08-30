from fastapi import HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from agents.weather_agent import WeatherAgent
from logger.bs_logger import bs_logger


class WeatherQuery(BaseModel):
    query: str
    city: str = "San Francisco"


class WeatherController:
    def __init__(self):
        self.weather_agent = WeatherAgent()
    
    async def get_sf_weather_with_agent(self, query: str = "What's the current weather in San Francisco?") -> Dict[str, Any]:
        """
        Get San Francisco weather using the LangChain agent
        
        Args:
            query (str): Weather query, defaults to asking about SF weather
            
        Returns:
            Dict[str, Any]: Agent response with weather information
        """
        try:
            bs_logger.info("Processing SF weather request with agent")
            result = await self.weather_agent.get_weather_response(query)
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result["error"])
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error in weather controller: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    def get_sf_weather_raw(self) -> Dict[str, Any]:
        """
        Get raw San Francisco weather data without agent processing
        
        Returns:
            Dict[str, Any]: Raw weather data
        """
        try:
            bs_logger.info("Getting raw SF weather data")
            return self.weather_agent.get_sf_weather()
            
        except Exception as e:
            bs_logger.error(f"Error getting raw weather data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get weather data: {str(e)}")
    
    async def query_weather(self, weather_query: WeatherQuery) -> Dict[str, Any]:
        """
        Process a custom weather query for any city
        
        Args:
            weather_query (WeatherQuery): Query object containing the question and optional city
            
        Returns:
            Dict[str, Any]: Agent response with weather information
        """
        try:
            bs_logger.info(f"Processing weather query for {weather_query.city}: {weather_query.query}")
            
            # Enhance the query with city information if not already included
            enhanced_query = weather_query.query
            if weather_query.city.lower() not in weather_query.query.lower():
                enhanced_query = f"{weather_query.query} for {weather_query.city}"
            
            result = await self.weather_agent.get_weather_response(enhanced_query)
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result["error"])
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error processing weather query: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")