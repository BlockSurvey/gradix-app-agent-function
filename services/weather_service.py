import requests
import json
from typing import Dict, Any
from logger.bs_logger import bs_logger


class WeatherService:
    def __init__(self):
        self.base_url = "http://wttr.in"
        
    def get_weather(self, city: str = "San Francisco") -> Dict[str, Any]:
        """
        Get current weather for a city using wttr.in API
        
        Args:
            city (str): City name, defaults to "San Francisco"
            
        Returns:
            Dict[str, Any]: Weather information
        """
        try:
            # Use wttr.in API which doesn't require API key
            url = f"{self.base_url}/{city}?format=j1"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            
            # Extract relevant information
            current = weather_data["current_condition"][0]
            location = weather_data["nearest_area"][0]
            
            formatted_data = {
                "location": {
                    "city": location["areaName"][0]["value"],
                    "region": location["region"][0]["value"],
                    "country": location["country"][0]["value"]
                },
                "current": {
                    "temperature_f": current["temp_F"],
                    "temperature_c": current["temp_C"],
                    "feels_like_f": current["FeelsLikeF"],
                    "feels_like_c": current["FeelsLikeC"],
                    "humidity": current["humidity"],
                    "description": current["weatherDesc"][0]["value"],
                    "wind_speed_mph": current["windspeedMiles"],
                    "wind_speed_kmh": current["windspeedKmph"],
                    "wind_direction": current["winddir16Point"],
                    "visibility_miles": current["visibilityMiles"],
                    "visibility_km": current["visibility"],
                    "uv_index": current["uvIndex"]
                },
                "observation_time": current["observation_time"]
            }
            
            return formatted_data
            
        except requests.exceptions.RequestException as e:
            bs_logger.error(f"Error fetching weather data: {e}")
            raise Exception(f"Failed to fetch weather data: {str(e)}")
        except (KeyError, IndexError) as e:
            bs_logger.error(f"Error parsing weather data: {e}")
            raise Exception(f"Failed to parse weather data: {str(e)}")
        except Exception as e:
            bs_logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error occurred: {str(e)}")