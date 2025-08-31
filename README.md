# Gradix Agent API Function

Gradix is an AI-powered application evaluation platform that automates candidate selection processes for events, accelerators, and other selection scenarios. This microservice provides AI agent functionalities to support the core Gradix platform with intelligent data processing and analysis capabilities.

## Features

- **AI-Powered Evaluation**: LangChain-based intelligent agents for automated decision-making
- **Multi-Format Processing**: Support for documents (PDF, DOCX), web content, and structured data
- **Cloud Integration**: Firebase Firestore for data persistence and Cloudflare R2 for file storage
- **Scalable Architecture**: FastAPI-based REST API with async support

## Setup Steps

### Prerequisites
- Python 3.11+
- Firebase service account credentials
- OpenAI API key (or Groq API key)
- Cloudflare R2 access credentials

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd gradix-app-agent-function
```

2. **Set up virtual environment**
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with the following required variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `GROQ_API_KEY`: Your Groq API key (optional, for alternative LLM provider)
- `FIREBASE_SERVICE_ACCOUNT_PATH`: Path to Firebase service account JSON
- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID
- `CLOUDFLARE_ACCESS_KEY`: R2 access key
- `CLOUDFLARE_SECRET_KEY`: R2 secret key
- `R2_BUCKET_NAME`: Your R2 bucket name (default: blocksurvey-ai-data-analysis)

5. **Run development server**
```bash
uvicorn app:app --reload
```

### Production Deployment

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐          │
│  │  Routes  │──│Controllers│──│   Agents    │          │
│  └──────────┘  └──────────┘  └─────────────┘          │
│       │             │              │                    │
│       └─────────────┼──────────────┘                    │
│                     │                                   │
│  ┌──────────────────┴────────────────────────┐         │
│  │              Services Layer               │         │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐ │        │
│  │  │Database │  │ Storage  │  │    AI     │ │        │
│  │  │Firebase │  │    R2    │  │  OpenAI   │ │        │
│  │  └─────────┘  └──────────┘  └──────────┘ │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### Core Components

- **`app.py`**: Main FastAPI application entry point
- **`config.py`**: Environment configuration management
- **`routes/`**: API endpoint definitions
  - `evaluation.py`: Evaluation-related endpoints
  - `dataset.py`: Dataset management endpoints
  - `rubric.py`: Rubric management endpoints
- **`controllers/`**: Business logic layer
  - `evaluation_controller.py`: Application evaluation logic
  - `dataset_controller.py`: Dataset operations
  - `rubric_controller.py`: Rubric handling
- **`agents/`**: LangChain AI agents
  - `application_evaluation_agent.py`: Main evaluation agent
  - `dataset_agent.py`: Dataset processing agent
  - `rubric_agent.py`: Rubric generation and management
- **`services/`**: External service integrations
  - `evaluation_service.py`: Evaluation processing
  - `weather_service.py`: Weather API integration (demo)
- **`database/`**: Firebase Firestore connectivity
- **`storage/`**: Cloudflare R2 S3-compatible storage
- **`logger/`**: Centralized logging configuration
- **`utils/`**: Utility functions and helpers

### Technology Stack

- **Web Framework**: FastAPI with Uvicorn ASGI server
- **Database**: Firebase Firestore (NoSQL)
- **Storage**: Cloudflare R2 (S3-compatible object storage)
- **AI/ML Framework**: LangChain with LangSmith tracing
- **LLM Providers**: OpenAI, Groq
- **Data Processing**: pandas, numpy
- **Document Processing**: PyPDF2, python-docx, beautifulsoup4
- **Authentication**: PyJWT for JWT token handling

## Models and Data

### AI Models

| Model | Provider | Version | Purpose | License |
|-------|----------|---------|---------|---------|
| GPT-4 | OpenAI | gpt-4 | Primary evaluation and reasoning | OpenAI Terms of Service |
| GPT-3.5-turbo | OpenAI | gpt-3.5-turbo | Fallback model for cost optimization | OpenAI Terms of Service |
| Mixtral-8x7b | Groq | mixtral-8x7b-32768 | Alternative high-performance model | Apache 2.0 |

### Data Processing

- **Input Formats**: CSV, JSON, PDF, DOCX, HTML
- **Output Formats**: JSON, structured evaluation reports
- **Storage**: Cloudflare R2 bucket for file persistence
- **Database**: Firebase Firestore for metadata and results

### Data Privacy and Compliance

- All data processing adheres to GDPR and CCPA guidelines
- User data is encrypted in transit and at rest
- No persistent storage of personally identifiable information (PII) without consent

## Known Limitations and Risks

### Technical Limitations

1. **Rate Limiting**: 
   - OpenAI API: 10,000 tokens/min (GPT-4), 90,000 tokens/min (GPT-3.5)
   - Firebase: 1 write/second per document
   - Cloudflare R2: 1000 requests/second

2. **File Size Constraints**:
   - Maximum PDF size: 50MB
   - Maximum CSV rows: 100,000
   - Maximum evaluation batch: 1000 applications

3. **Processing Time**:
   - Average evaluation time: 5-30 seconds per application
   - Batch processing may take several minutes

### Operational Risks

1. **API Dependency**: Service availability depends on third-party APIs (OpenAI, Firebase, Cloudflare)
2. **Cost Management**: LLM usage can incur significant costs at scale
3. **Model Hallucination**: AI responses may occasionally generate incorrect information
4. **Bias Mitigation**: Ongoing monitoring required for fairness in evaluation

### Security Considerations

- Regular security audits recommended
- API keys must be rotated periodically
- Input validation required for all user-submitted content
- Rate limiting implemented to prevent abuse

## API Endpoints

### Core Endpoints

- `GET /` - Health check and API information
- `POST /evaluate/application` - Evaluate single application
- `POST /evaluate/batch` - Batch evaluation processing
- `GET /evaluate/status/{job_id}` - Check evaluation job status
- `POST /rubric/generate` - Generate evaluation rubric
- `POST /dataset/upload` - Upload dataset for processing
- `GET /dataset/{dataset_id}` - Retrieve dataset information

### Development/Testing

- `GET /users/{user_id}` - Get user information (demo)
- `POST /weather/sf` - Weather API integration demo

## Team and Contacts

### Core Team

- **Project Lead**: Gradix Development Team
- **Organization**: BlockSurvey
- **Repository**: Gradix App Agent Function

### Contact Information

- **Technical Support**: [support@blocksurvey.io]
- **Bug Reports**: [GitHub Issues](https://github.com/blocksurvey/gradix/issues)
- **Security Issues**: [security@blocksurvey.io]
- **Documentation**: [Gradix Documentation](https://docs.gradix.io)

### Contributing

This project is part of the Gradix platform by BlockSurvey. We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- All tests pass
- Code follows PEP 8 style guidelines
- Documentation is updated
- Commit messages are descriptive

## License

This project is proprietary software owned by BlockSurvey. All rights reserved.

For licensing inquiries, please contact [legal@blocksurvey.io].

## Acknowledgments

- LangChain community for excellent AI orchestration tools
- OpenAI for providing state-of-the-art language models
- Firebase and Cloudflare for reliable cloud infrastructure
- FastAPI team for the high-performance web framework

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Active Development