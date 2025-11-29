"""
Main entry point for the quiz solver.
"""

import uvicorn
from .config import settings
from .api import app


def main():
    """Run the quiz solver API server."""
    uvicorn.run(
        "quiz_solver.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
