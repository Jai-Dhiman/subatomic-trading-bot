"""
FastAPI application for energy trading predictions.

Serves transformer models through REST API endpoints.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.inference import get_model_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Loads models on startup and cleans up on shutdown.
    """
    logger.info("Starting FastAPI application...")
    logger.info("Loading transformer models...")
    
    try:
        manager = get_model_manager()
        manager.load_models()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    yield
    
    logger.info("Shutting down FastAPI application...")


app = FastAPI(
    title="Energy Trading API",
    description="AI-powered energy trading predictions using dual transformer models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Energy Trading API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
