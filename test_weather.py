#!/usr/bin/env python3
"""
Simple test script for the weather service functionality
Run this to test the weather service without needing OpenAI API key
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.weather_service import WeatherService
from logger.bs_logger import bs_logger


async def test_weather_service():
    """Test the basic weather service functionality"""
    print("🌤️  Testing Gradix Weather Service")
    print("=" * 50)
    
    weather_service = WeatherService()
    
    try:
        # Test San Francisco weather
        print("📍 Getting weather for San Francisco...")
        sf_weather = weather_service.get_weather("San Francisco")
        
        location = sf_weather["location"]
        current = sf_weather["current"]
        
        print(f"\n🌍 Location: {location['city']}, {location['region']}, {location['country']}")
        print(f"🌡️  Temperature: {current['temperature_f']}°F ({current['temperature_c']}°C)")
        print(f"🤔 Feels like: {current['feels_like_f']}°F ({current['feels_like_c']}°C)")
        print(f"☁️  Condition: {current['description']}")
        print(f"💧 Humidity: {current['humidity']}%")
        print(f"💨 Wind: {current['wind_speed_mph']} mph {current['wind_direction']}")
        print(f"👁️  Visibility: {current['visibility_miles']} miles")
        print(f"☀️  UV Index: {current['uv_index']}")
        print(f"⏰ Observation time: {sf_weather['observation_time']}")
        
        print("\n✅ Weather service is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing weather service: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 Starting Gradix Weather Service Tests\n")
    
    success = await test_weather_service()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! The weather service is ready.")
        print("\nNext steps:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Start the server: uvicorn app:app --reload")
        print("3. Test the agent endpoint: POST /weather/sf")
    else:
        print("❌ Tests failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)