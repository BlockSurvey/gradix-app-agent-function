from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from database.firebase_db import firestore_client
from routes.main import api_router

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

# Include all API routes
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "Gradix Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "api": {
            "version": "v1",
            "prefix": "/api/v1"
        }
    }