"""
FastAPI application for the quiz solver API.
"""

import asyncio
import os
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError

from .config import settings
from .models import QuizRequest, QuizResponse, ErrorResponse
from .pipeline import generate_quiz_id, run_quiz_background
from .logging_utils import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Quiz Solver API starting up...")
    logger.info(f"Configured secrets for {len(settings.valid_secrets)} students")
    yield
    logger.info("Quiz Solver API shutting down...")


app = FastAPI(
    title="LLM Quiz Solver API",
    description="Automated quiz solving agent with 3-minute deadline support",
    version="0.1.0",
    lifespan=lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle invalid JSON payloads."""
    logger.warning(f"Invalid request: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid JSON payload"}
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "quiz-solver"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/logs")
async def get_logs(
    lines: int = Query(default=100, ge=1, le=5000, description="Number of lines to return"),
    level: str = Query(default="all", description="Filter by log level (all, info, warning, error)")
):
    """
    Retrieve recent log entries.
    
    - **lines**: Number of log lines to return (default: 100, max: 5000)
    - **level**: Filter by log level (all, info, warning, error)
    """
    log_dir = os.environ.get("LOG_DIR", "/app/logs")
    log_file = os.path.join(log_dir, "quiz_solver.log")
    
    # Fallback to current directory
    if not os.path.exists(log_file):
        log_file = "quiz_solver.log"
    
    if not os.path.exists(log_file):
        return PlainTextResponse("No log file found", status_code=404)
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        # Filter by level if specified
        if level.lower() != "all":
            level_upper = level.upper()
            all_lines = [line for line in all_lines if level_upper in line]
        
        # Get last N lines
        recent_lines = all_lines[-lines:]
        
        return PlainTextResponse("".join(recent_lines))
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return PlainTextResponse(f"Error reading logs: {e}", status_code=500)


@app.delete("/logs")
async def clear_logs():
    """Clear the log file."""
    log_dir = os.environ.get("LOG_DIR", "/app/logs")
    log_file = os.path.join(log_dir, "quiz_solver.log")
    
    if not os.path.exists(log_file):
        log_file = "quiz_solver.log"
    
    try:
        if os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("")
            logger.info("Log file cleared")
            return {"status": "ok", "message": "Log file cleared"}
        else:
            return {"status": "ok", "message": "No log file to clear"}
    except Exception as e:
        logger.error(f"Error clearing log file: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing logs: {e}")


@app.post("/api/quiz", response_model=QuizResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid JSON payload"},
    403: {"model": ErrorResponse, "description": "Invalid email or secret"}
})
async def handle_quiz_request(request: QuizRequest):
    """
    Main API endpoint for quiz requests.
    
    Validates request, initiates quiz solving pipeline.
    
    - **email**: Student email address
    - **secret**: Authentication secret
    - **url**: Quiz URL to solve
    """
    
    try:
        # 1. VALIDATE REQUEST FIELDS
        if not request.email or not request.secret or not request.url:
            logger.warning(f"Missing required fields in request")
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # 2. VERIFY SECRET
        if request.email not in settings.valid_secrets:
            logger.warning(f"Invalid email attempt: {request.email}")
            raise HTTPException(status_code=403, detail="Invalid email or secret")
        
        if settings.valid_secrets[request.email] != request.secret:
            logger.warning(f"Invalid secret for {request.email}")
            raise HTTPException(status_code=403, detail="Invalid email or secret")
        
        # 3. LOG REQUEST
        logger.info(f"Valid request from {request.email} for {request.url}")
        
        # 4. GENERATE RESPONSE
        quiz_id = generate_quiz_id(request.url)
        response = QuizResponse(
            status="accepted",
            quiz_id=quiz_id,
            timestamp=datetime.now().isoformat()
        )
        
        # 5. LAUNCH ASYNC QUIZ SOLVER (don't wait for completion)
        asyncio.create_task(
            run_quiz_background(
                email=request.email,
                secret=request.secret,
                url=request.url
            )
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app
