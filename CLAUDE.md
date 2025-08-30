# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Setup and Installation:**
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env file with your API keys
```

**Test Weather Service:**
```bash
python test_weather.py
```

**Development Server:**
```bash
uvicorn app:app --reload
```

**With specific Python version:**
```bash
python3.11 -m uvicorn app:app --reload
```

**Virtual Environment Setup:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Production Server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Architecture Overview

This is a FastAPI-based Python application with a modular architecture:

**Core Structure:**
- `app.py` - Main FastAPI application entry point with weather and user routes
- `config.py` - Environment configuration management using python-dotenv
- `controllers/` - Business logic layer (UserController, WeatherController)
- `agents/` - LangChain AI agents (WeatherAgent for intelligent weather queries)
- `services/` - External service integrations (WeatherService for wttr.in API)
- `database/` - Database connectivity (Firebase Firestore integration)
- `logger/` - Centralized logging configuration using Python's logging module
- `storage/` - External storage integration (Cloudflare R2 via boto3)
- `utils/` - Utility functions and helpers

**Key Dependencies:**
- **Web Framework:** FastAPI with Uvicorn server
- **Database:** Firebase Firestore via firebase-admin SDK
- **Storage:** Cloudflare R2 via boto3 S3-compatible client
- **AI/ML:** LangChain ecosystem (langchain, langchain-openai, langchain-groq)
- **Data Processing:** pandas, numpy for data manipulation
- **Authentication:** JWT support via PyJWT
- **Document Processing:** PyPDF2, python-docx, beautifulsoup4

**Environment Configuration:**
- Uses .env file for environment variables
- Key configurations include LLM models (GPT-3.5-turbo, GPT-4), Cloudflare R2 credentials, and Firebase service account
- Default LLM model is GPT-4, with GPT-3.5-turbo as fallback

**Database Integration:**
- Firebase Firestore client initialized in `database/firebase_db.py`
- Service account authentication via JSON key file

**Storage Integration:**
- Cloudflare R2 bucket operations via S3-compatible API
- CSV file reading/writing capabilities with pandas integration
- Specific bucket: 'blocksurvey-ai-data-analysis'

**Logging:**
- Configured for INFO level with timestamp formatting
- Console output (file logging commented out)
- Centralized logger instance: `bs_logger`

## Project Context

This appears to be a FastAPI microservice for a DApp (decentralized application) called Gradix, specifically functioning as an agent service with AI/ML capabilities for data analysis and processing.