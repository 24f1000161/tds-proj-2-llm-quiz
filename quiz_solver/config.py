"""
Configuration management for the quiz solver.
"""

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration with token budget management."""
    
    # Aipipe settings (OpenAI-compatible via aipipe.org)
    aipipe_token: Optional[str] = None
    aipipe_base_url: str = "https://aipipe.org/openai/v1"
    aipipe_model: str = "gpt-4o-mini"
    aipipe_monthly_tokens: int = 1_000_000
    
    # Gemini fallback settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    gemini_via_aipipe: bool = True  # Use Gemini through aipipe proxy
    
    # Token thresholds
    auto_switch_threshold: float = 0.75
    emergency_bypass: float = 0.90


class TimeoutConfig(BaseModel):
    """Timeout configuration."""
    
    quiz_deadline_seconds: int = 180  # 3 minutes as required
    safety_buffer_seconds: int = 10  # Reserve for final submission
    llm_timeout_primary: int = 30
    llm_timeout_fallback: int = 30
    navigation_timeout: int = 30000  # 30 seconds in ms
    download_timeout: int = 30


class BrowserConfig(BaseModel):
    """Browser configuration for Playwright."""
    
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class Settings(BaseModel):
    """Main application settings."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Student secrets (loaded from environment or config)
    valid_secrets: dict[str, str] = {}
    
    # Sub-configurations
    llm: LLMConfig = LLMConfig()
    timeouts: TimeoutConfig = TimeoutConfig()
    browser: BrowserConfig = BrowserConfig()
    
    # Logging
    log_level: str = "INFO"


def load_settings() -> Settings:
    """Load settings from environment variables."""
    
    # Load student secrets from environment
    # Format: STUDENT_SECRETS=email1:secret1,email2:secret2
    secrets_str = os.getenv("STUDENT_SECRETS", "")
    valid_secrets = {}
    if secrets_str:
        for pair in secrets_str.split(","):
            if ":" in pair:
                email, secret = pair.split(":", 1)
                valid_secrets[email.strip()] = secret.strip()
    
    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        valid_secrets=valid_secrets,
        llm=LLMConfig(
            aipipe_token=os.getenv("AIPIPE_TOKEN"),
            aipipe_base_url=os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1"),
            aipipe_model=os.getenv("AIPIPE_MODEL", "gpt-4o-mini"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            gemini_via_aipipe=os.getenv("GEMINI_VIA_AIPIPE", "true").lower() == "true",
        ),
        timeouts=TimeoutConfig(
            quiz_deadline_seconds=int(os.getenv("QUIZ_DEADLINE_SECONDS", "180")),
        ),
        browser=BrowserConfig(
            headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true",
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


# Global settings instance
settings = load_settings()
