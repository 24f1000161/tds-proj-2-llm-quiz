"""
LLM client management with dual-model strategy.
Supports aipipe (GPT models via OpenAI-compatible API) and Gemini as fallback.
Uses httpx for all API calls.
"""

import json
from typing import Optional, Any
from dataclasses import dataclass, field
import httpx

from .config import settings
from .logging_utils import logger


@dataclass
class TokenTracker:
    """Track token usage across models."""
    
    aipipe_used: int = 0
    aipipe_total: int = field(default_factory=lambda: settings.llm.aipipe_monthly_tokens)
    gemini_used: int = 0
    
    @property
    def aipipe_usage_pct(self) -> float:
        """Get aipipe usage percentage."""
        return self.aipipe_used / self.aipipe_total if self.aipipe_total > 0 else 1.0
    
    def should_use_gemini(self) -> bool:
        """Determine if we should use Gemini based on token budget."""
        return self.aipipe_usage_pct >= settings.llm.auto_switch_threshold


class LLMClient:
    """Unified LLM client with automatic model switching."""
    
    def __init__(self):
        self.token_tracker = TokenTracker()
        self._aipipe_configured = False
        self._gemini_configured = False
        self._http_client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self) -> None:
        """Initialize LLM clients."""
        
        # Create shared httpx client
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeouts.llm_timeout_primary)
        )
        
        # Check aipipe configuration
        if settings.llm.aipipe_token:
            self._aipipe_configured = True
            logger.info(f"Aipipe configured with base URL: {settings.llm.aipipe_base_url}")
        else:
            logger.warning("No AIPIPE_TOKEN found, skipping aipipe initialization")
        
        # Check Gemini configuration
        if settings.llm.gemini_via_aipipe and settings.llm.aipipe_token:
            self._gemini_configured = True
            logger.info("Gemini configured via aipipe proxy")
        elif settings.llm.gemini_api_key:
            self._gemini_configured = True
            logger.info("Gemini configured with direct API key")
        else:
            logger.warning("No Gemini configuration found")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
    
    def _select_model(self) -> str:
        """Select which model to use based on token budget and availability.
        
        Since aipipe OpenAI quota is often exhausted, we prefer Gemini via aipipe
        which uses the native Gemini endpoint.
        """
        
        # Prefer Gemini as primary since aipipe OpenAI quota is often exhausted
        if self._gemini_configured:
            logger.debug("Using Gemini via aipipe (preferred)")
            return "gemini"
        
        if self._aipipe_configured:
            logger.debug("Using aipipe (fallback)")
            return "aipipe"
        
        raise RuntimeError("No LLM client available")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        json_response: bool = False
    ) -> str:
        """Generate a response from the LLM with retry logic."""
        import asyncio
        
        model_choice = self._select_model()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if model_choice == "aipipe":
                    return await self._generate_aipipe(prompt, max_tokens, temperature, json_response)
                else:
                    return await self._generate_gemini(prompt, max_tokens, temperature)
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limiting with retry
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Primary model ({model_choice}) failed: {e}")
                
                # Try fallback on final failure
                if model_choice == "aipipe" and self._gemini_configured:
                    logger.info("Falling back to Gemini")
                    try:
                        return await self._generate_gemini(prompt, max_tokens, temperature)
                    except Exception as fallback_e:
                        logger.error(f"Fallback to Gemini also failed: {fallback_e}")
                        raise
                elif model_choice == "gemini" and self._aipipe_configured:
                    logger.info("Falling back to aipipe")
                    try:
                        return await self._generate_aipipe(prompt, max_tokens, temperature, json_response)
                    except Exception as fallback_e:
                        logger.error(f"Fallback to aipipe also failed: {fallback_e}")
                        raise
                raise
        
        raise RuntimeError(f"All {max_retries} attempts failed")
    
    async def _generate_aipipe(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        json_response: bool
    ) -> str:
        """Generate using aipipe via OpenRouter endpoint.
        
        Uses: https://aipipe.org/openrouter/v1/chat/completions
        Models: openai/gpt-4.1-nano, google/gemini-2.0-flash-lite, etc.
        """
        
        if not self._aipipe_configured or not self._http_client:
            raise RuntimeError("Aipipe client not initialized")
        
        # Use OpenRouter endpoint for all models via aipipe
        url = "https://aipipe.org/openrouter/v1/chat/completions"
        
        # Model needs provider prefix for OpenRouter
        model = settings.llm.aipipe_model
        if not "/" in model:
            # Add openai/ prefix if not present
            model = f"openai/{model}"
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        
        headers = {
            "Authorization": f"Bearer {settings.llm.aipipe_token}",
            "Content-Type": "application/json"
        }
        
        logger.debug(f"Calling OpenRouter via aipipe: {url} with model {model}")
        response = await self._http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Track token usage
        if "usage" in data:
            self.token_tracker.aipipe_used += data["usage"].get("total_tokens", 0)
        
        content = data["choices"][0]["message"]["content"]
        logger.debug(f"Aipipe response: {content[:100]}...")
        
        return content or ""
    
    async def _generate_gemini(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using Gemini (via aipipe or direct API)."""
        
        if not self._gemini_configured or not self._http_client:
            raise RuntimeError("Gemini client not initialized")
        
        if settings.llm.gemini_via_aipipe and settings.llm.aipipe_token:
            # Use Gemini via aipipe's OpenRouter proxy
            return await self._generate_gemini_via_aipipe(prompt, max_tokens, temperature)
        else:
            # Use direct Gemini API
            return await self._generate_gemini_direct(prompt, max_tokens, temperature)
    
    async def _generate_gemini_via_aipipe(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using Gemini via aipipe's native Gemini endpoint.
        
        Uses: https://aipipe.org/geminiv1beta/models/MODEL:generateContent
        Auth: Authorization: Bearer {AIPIPE_TOKEN}
        """
        
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        
        # Use aipipe's native Gemini endpoint
        # Model format: gemini-2.0-flash, gemini-1.5-flash, etc.
        model = settings.llm.gemini_model
        # Remove any prefix if present
        if model.startswith("google/"):
            model = model[7:]
        
        url = f"https://aipipe.org/geminiv1beta/models/{model}:generateContent"
        
        # Gemini native request format
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        # Use Authorization Bearer header for aipipe (same as OpenRouter)
        headers = {
            "Authorization": f"Bearer {settings.llm.aipipe_token}",
            "Content-Type": "application/json"
        }
        
        logger.debug(f"Calling Gemini via aipipe: {url}")
        response = await self._http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Track token usage from Gemini response
        if "usageMetadata" in data:
            self.token_tracker.gemini_used += data["usageMetadata"].get("totalTokenCount", 0)
        
        # Parse Gemini response format with robust error handling
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                error_msg = data.get("error", {}).get("message", "No candidates in response")
                raise ValueError(f"Gemini API error: {error_msg}")
            
            content_obj = candidates[0].get("content", {})
            parts = content_obj.get("parts", [])
            if not parts:
                # Sometimes Gemini returns empty parts, check for other fields
                finish_reason = candidates[0].get("finishReason", "")
                if finish_reason == "SAFETY":
                    raise ValueError("Response blocked by safety filters")
                raise ValueError("No parts in response content")
            
            content = parts[0].get("text", "")
            if not content:
                raise ValueError("Empty text in response")
            
            logger.debug(f"Gemini (via aipipe) response: {content[:100]}...")
            return content
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse Gemini response: {e}, data: {data}")
            raise ValueError(f"Invalid Gemini response structure: {e}")
    
    async def _generate_gemini_direct(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using direct Gemini API."""
        
        if not self._http_client or not settings.llm.gemini_api_key:
            raise RuntimeError("Gemini direct API not configured")
        
        # Use Gemini's generateContent API
        model = settings.llm.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {
            "x-goog-api-key": settings.llm.gemini_api_key,
            "Content-Type": "application/json"
        }
        
        response = await self._http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Track token usage
        if "usageMetadata" in data:
            self.token_tracker.gemini_used += data["usageMetadata"].get("totalTokenCount", 0)
        
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        logger.debug(f"Gemini (direct) response: {content[:100]}...")
        
        return content or ""
    
    async def parse_question(self, question_text: str) -> dict[str, Any]:
        """Parse a question using LLM."""
        
        parse_prompt = f"""Parse this quiz question into structured format. Extract ONLY valid JSON:

QUESTION: {question_text}

Return JSON (no other text):
{{
  "task_type": "sourcing|cleansing|analysis|visualization|multi_step",
  "primary_objective": "<one sentence summary>",
  "data_sources": [<list of URLs or API endpoints>],
  "required_steps": [<list of 2-3 processing steps>],
  "answer_format": "number|string|boolean|json|base64_image",
  "precision_required": "<e.g. '2 decimal places' or null>",
  "submit_endpoint": "<submit URL if mentioned>",
  "key_constraints": [<list of constraints>]
}}"""
        
        try:
            response = await self.generate(parse_prompt, max_tokens=500, json_response=True)
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return {}
    
    async def classify_task(self, question_text: str) -> dict[str, Any]:
        """Classify a task using LLM."""
        
        classify_prompt = f"""Analyze this quiz question and classify what needs to be done.

QUESTION: {question_text}

Return ONLY JSON with this structure:
{{
  "task_type": "sourcing|cleansing|analysis|visualization|multi_step|scraping",
  "data_formats": ["pdf", "csv", "json", "html", "api"],
  "data_sources": ["<any URLs mentioned that need to be fetched>"],
  "scrape_action": {{
    "needed": true/false,
    "url": "<relative or absolute URL to scrape if mentioned>",
    "target": "<what to extract from the page>"
  }},
  "analysis_required": "aggregation|filtering|statistical|text_extraction|none",
  "complexity": 1-5,
  "expected_answer_type": "number|string|boolean|json|base64_image",
  "processing_steps": [
    {{"step": 1, "action": "<action>", "description": "<what to do>"}}
  ]
}}

IMPORTANT: If the question mentions scraping a URL or getting data from a page, set scrape_action.needed=true and provide the URL.
Look for patterns like "Scrape <url>", "Get data from <url>", "Visit <url>", etc."""
        
        try:
            response = await self.generate(classify_prompt, max_tokens=600, json_response=True)
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            result = json.loads(response)
            logger.info(f"LLM classification result: {result}")
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse classification response as JSON")
            return {
                "task_type": "multi_step",
                "complexity": 3,
                "processing_steps": []
            }
    
    async def generate_analysis_code(
        self,
        df_info: dict[str, Any],
        question_text: str
    ) -> str:
        """Generate pandas analysis code."""
        
        # Build additional context if available
        additional_context = ""
        if df_info.get('additional_context'):
            additional_context = f"\nADDITIONAL CONTEXT:\n{df_info.get('additional_context')}"
        
        analysis_prompt = f"""You are a pandas expert. Generate ONLY executable Python code.

CRITICAL: The dataframe 'df' is ALREADY LOADED in memory. DO NOT read CSV files. DO NOT use requests.

AVAILABLE:
- Variable 'df' contains the data (already loaded)
- 'pandas as pd' (already imported)
- 'numpy as np' (already imported)

QUESTION: {question_text}
DATAFRAME SHAPE: {df_info.get('shape', 'unknown')}
COLUMNS: {df_info.get('columns', [])}
DTYPES: {df_info.get('dtypes', {})}
SAMPLE DATA (first 5 rows):
{df_info.get('sample', '')}
{additional_context}

REQUIREMENTS:
- DO NOT import anything - pandas and numpy are already imported
- DO NOT read files - df is already loaded with the data
- Use the existing 'df' variable directly
- Store the final answer in variable 'answer'
- For numeric answers: use int(answer) to ensure Python int type
- If AUDIO TRANSCRIPT mentions "greater than or equal" use >= operator
- If AUDIO TRANSCRIPT mentions "greater than" use > operator
- Read the AUDIO TRANSCRIPT or CUTOFF VALUE carefully for the filter condition

Generate ONLY 2-3 lines of Python code (no imports, no file reading):
"""
        
        response = await self.generate(analysis_prompt, max_tokens=500, temperature=0.1)
        
        # Clean up code block markers if present
        code = response.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
    async def answer_from_context(self, context: str, question_text: str) -> str:
        """Answer a question directly from provided context (multi-modal fusion)."""
        
        answer_prompt = f"""Based on the following context, answer the question precisely.

QUESTION: {question_text}

CONTEXT:
{context[:8000]}

Instructions:
- Provide ONLY the answer, nothing else
- If the answer is a number, return just the number
- If the answer is a list, return comma-separated values
- Be precise and concise

Answer:"""
        
        response = await self.generate(answer_prompt, max_tokens=500, temperature=0.1)
        return response.strip()


# Global LLM client instance
llm_client = LLMClient()
