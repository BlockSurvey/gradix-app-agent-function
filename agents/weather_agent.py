from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from langchain import hub
from typing import Dict, Any
import json

from services.weather_service import WeatherService
from logger.bs_logger import bs_logger
from config import LLM_MODEL_DEFAULT


class WeatherAgent:
    def __init__(self):
        self.weather_service = WeatherService()
        self.llm = ChatOpenAI(
            model=LLM_MODEL_DEFAULT,
            temperature=0
        )
        self._setup_agent()
    
    def _get_weather_tool(self) -> Tool:
        """Create a weather tool for the agent"""
        def get_weather(city: str) -> str:
            """Get current weather information for a city"""
            try:
                weather_data = self.weather_service.get_weather(city)
                return json.dumps(weather_data, indent=2)
            except Exception as e:
                return f"Error getting weather for {city}: {str(e)}"
        
        return Tool(
            name="get_weather",
            description="Get current weather information for any city. Use this when asked about weather conditions, temperature, or meteorological data for a specific location.",
            func=get_weather
        )
    
    def _setup_agent(self):
        """Set up the LangChain agent with weather tools"""
        tools = [self._get_weather_tool()]
        
        # Get the prompt from the hub or create a custom one
        try:
            prompt = hub.pull("hwchase17/openai-functions-agent")
        except:
            # Fallback prompt if hub is not accessible
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a helpful AI assistant that can provide weather information for any city.
                
When asked about weather:
1. Use the get_weather tool to fetch current weather data
2. Provide a comprehensive but concise summary of the weather conditions
3. Include temperature (both Fahrenheit and Celsius), weather description, humidity, wind conditions, and any other relevant details
4. Be conversational and friendly in your response
5. If asked about San Francisco specifically, make sure to use the exact city name

Always use the available tools to get real-time data rather than relying on your training data."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        
        # Create the agent
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            return_intermediate_steps=True
        )
    
    async def get_weather_response(self, query: str) -> Dict[str, Any]:
        """
        Process a weather query and return the agent's response
        
        Args:
            query (str): The weather-related query
            
        Returns:
            Dict[str, Any]: Response containing the weather information and agent reasoning
        """
        try:
            bs_logger.info(f"Processing weather query: {query}")
            
            result = await self.agent_executor.ainvoke({"input": query})
            
            return {
                "success": True,
                "query": query,
                "response": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "agent_type": "weather_agent"
            }
            
        except Exception as e:
            bs_logger.error(f"Error processing weather query: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "agent_type": "weather_agent"
            }
    
    def get_sf_weather(self) -> Dict[str, Any]:
        """
        Get current weather for San Francisco specifically
        
        Returns:
            Dict[str, Any]: Weather data for San Francisco
        """
        try:
            return self.weather_service.get_weather("San Francisco")
        except Exception as e:
            bs_logger.error(f"Error getting SF weather: {e}")
            raise