"""
Logging utilities for the quiz solver.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Any
from logging.handlers import RotatingFileHandler

from .config import settings


def setup_logging() -> logging.Logger:
    """Configure logging for the application with console and file handlers."""
    
    logger = logging.getLogger("quiz_solver")
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # File handler (for Hugging Face Spaces persistence)
    # HF Spaces: /tmp is always writable, /app/logs may not be
    log_dir = os.environ.get("LOG_DIR", "/tmp/quiz_solver_logs")
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            log_dir = "/tmp"  # Fallback to /tmp which is always writable
    
    log_file = os.path.join(log_dir, "quiz_solver.log")
    file_handler = None
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        print(f"📝 Logging to file: {log_file}")
    except Exception as e:
        file_handler = None
        print(f"Warning: Could not create file handler: {e}")
    
    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
        if file_handler:
            logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logging()


def log_step(session: Any, step_name: str, details: dict[str, Any]) -> None:
    """Log a pipeline step to the session audit trail."""
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "details": details
    }
    
    if hasattr(session, 'audit_trail'):
        session.audit_trail.append(entry)
    
    logger.info(f"[{step_name}] {details}")
