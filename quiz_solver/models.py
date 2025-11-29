"""
Pydantic models for the quiz solver.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class QuizRequest(BaseModel):
    """Incoming quiz request from API endpoint."""
    
    email: str
    secret: str
    url: str


class QuizResponse(BaseModel):
    """Response to quiz request."""
    
    status: str = "accepted"
    quiz_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str


class TaskType(str, Enum):
    """Types of quiz tasks."""
    
    SOURCING = "sourcing"
    CLEANSING = "cleansing"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    MULTI_STEP = "multi_step"


class AnswerFormat(str, Enum):
    """Expected answer formats."""
    
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    JSON = "json"
    BASE64_IMAGE = "base64_image"


class QuestionComponents(BaseModel):
    """Parsed question components."""
    
    question_id: Optional[str] = None
    question_description: Optional[str] = None
    data_sources: list[str] = Field(default_factory=list)
    relative_urls: list[str] = Field(default_factory=list)  # Relative URLs like /path/to/data
    instructions: list[str] = Field(default_factory=list)
    submit_url: Optional[str] = None
    relative_submit_url: Optional[str] = None  # Relative submit URL like /submit
    answer_format: AnswerFormat = AnswerFormat.STRING


class TaskClassification(BaseModel):
    """LLM-generated task classification."""
    
    task_type: TaskType = TaskType.MULTI_STEP
    data_formats: list[str] = Field(default_factory=list)
    analysis_required: Optional[str] = None
    complexity: int = 3
    expected_answer_type: AnswerFormat = AnswerFormat.STRING
    estimated_tokens: int = 300
    processing_steps: list[dict[str, Any]] = Field(default_factory=list)


class SessionState(BaseModel):
    """Session state for quiz solving."""
    
    # Request info
    email: str
    secret: str
    initial_url: str
    
    # Timing
    start_time: float
    deadline: float
    deadline_safety_buffer: int = 10
    
    # Quiz tracking
    quiz_id: Optional[str] = None
    question_text: Optional[str] = None
    question_id: Optional[str] = None
    submit_url: Optional[str] = None
    expected_format: Optional[AnswerFormat] = None
    
    # Data & Analysis
    data_sources: list[str] = Field(default_factory=list)
    raw_data: Optional[Any] = None
    cleaned_data: Optional[Any] = None
    analysis_result: Optional[Any] = None
    final_answer: Optional[Any] = None
    
    # Attempt tracking
    current_url: str
    quiz_chain: list[str] = Field(default_factory=list)
    submission_attempts: list[dict[str, Any]] = Field(default_factory=list)
    
    # Logs
    llm_calls_log: list[dict[str, Any]] = Field(default_factory=list)
    error_log: list[str] = Field(default_factory=list)
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class SubmissionResult(BaseModel):
    """Result from quiz submission."""
    
    correct: bool = False
    reason: Optional[str] = None
    url: Optional[str] = None  # Next quiz URL if chained
    message: Optional[str] = None
