# Gradix Agent API Function

Gradix is an AI-powered application evaluation platform that automates candidate selection processes for events, accelerators, and other selection scenarios. This microservice provides AI agent functionalities to support the core Gradix platform with intelligent data processing and analysis capabilities.

## Features

- **AI-Powered Evaluation**: LangChain-based intelligent agents for automated decision-making
- **Multi-Format Processing**: Support for documents (PDF, DOCX), web content, and structured data
- **Cloud Integration**: Firebase Firestore for data persistence and Cloudflare R2 for file storage
- **Scalable Architecture**: FastAPI-based REST API with async support

## Quick Start

### Prerequisites
- Python 3.11+
- Firebase service account credentials
- OpenAI API key
- Cloudflare R2 access credentials

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file:
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Development

Start the development server:
```bash
uvicorn app:app --reload
```

Run with specific Python version:
```bash
python3.11 -m uvicorn app:app --reload
```

### Virtual Environment Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Production Deployment

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## API Endpoints

- `GET /` - Health check endpoint
- `GET /users/{user_id}` - Get user information
- `POST /weather/sf` - Get current weather in San Francisco using AI agent

## Architecture

- **FastAPI**: High-performance async web framework
- **LangChain**: AI agent orchestration and LLM integration
- **Firebase**: Cloud-based NoSQL database
- **Cloudflare R2**: S3-compatible object storage
- **OpenAI/Groq**: Large language model providers

## Contributing

This project is part of the Gradix platform by BlockSurvey. For contribution guidelines and development setup, please refer to the main project documentation.