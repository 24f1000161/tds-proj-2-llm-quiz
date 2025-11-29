"""
Answer submission module.
"""

from typing import Any
import aiohttp
import json
import numpy as np

from .config import settings
from .models import SubmissionResult
from .logging_utils import logger, log_step


def convert_to_json_serializable(value: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.bool_):
        return bool(value)
    return value


async def submit_answer(
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
    submit_url: str | None,
    session: Any
) -> SubmissionResult:
    """Submit answer to quiz endpoint."""
    from urllib.parse import urljoin, urlparse
    
    # Convert numpy types to native Python types
    answer = convert_to_json_serializable(answer)
    
    # If no submit URL provided, use standard /submit relative to base
    if not submit_url:
        # Get base URL (scheme + netloc)
        parsed = urlparse(quiz_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        submit_url = f"{base_url}/submit"
        logger.info(f"No submit URL provided, using default: {submit_url}")
    elif submit_url.startswith('/'):
        # Relative URL - resolve against quiz URL base
        parsed = urlparse(quiz_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        submit_url = f"{base_url}{submit_url}"
        logger.info(f"Resolved relative submit URL to: {submit_url}")
    
    logger.info(f"Submitting answer to {submit_url}")
    logger.info(f"Answer: {answer}")
    
    # Validate answer size (< 1MB)
    answer_str = str(answer) if not isinstance(answer, str) else answer
    if len(answer_str.encode('utf-8')) > 1_000_000:
        logger.error("Answer exceeds 1MB size limit")
        return SubmissionResult(
            correct=False,
            reason="Answer exceeds 1MB size limit"
        )
    
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=settings.timeouts.download_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as http_session:
            async with http_session.post(
                submit_url,
                json=payload,
                ssl=False
            ) as resp:
                response_text = await resp.text()
                
                log_step(session, "answer_submitted", {
                    "submit_url": submit_url,
                    "answer": str(answer)[:100],
                    "status": resp.status,
                    "response": response_text[:200]
                })
                
                if resp.status == 200:
                    try:
                        response_data = json.loads(response_text)
                        return SubmissionResult(
                            correct=response_data.get('correct', False),
                            reason=response_data.get('reason'),
                            url=response_data.get('url'),
                            message=response_data.get('message')
                        )
                    except json.JSONDecodeError:
                        # Check for success indicators in text
                        if 'correct' in response_text.lower() or 'success' in response_text.lower():
                            return SubmissionResult(correct=True, message=response_text)
                        return SubmissionResult(correct=False, message=response_text)
                else:
                    return SubmissionResult(
                        correct=False,
                        reason=f"HTTP {resp.status}",
                        message=response_text
                    )
    
    except Exception as e:
        logger.error(f"Submission failed: {e}")
        return SubmissionResult(
            correct=False,
            reason=str(e)
        )
