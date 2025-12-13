"""
LLM client management with dual-model strategy.

Model routing:
- OpenAI models (gpt-4o-mini, etc.): Use AIPIPE proxy only
- Gemini models: Use DIRECT Google API (GEMINI_API_KEY)

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
        
        # Check Gemini configuration - PREFER DIRECT API
        if settings.llm.gemini_api_key:
            self._gemini_configured = True
            logger.info(f"Gemini configured with DIRECT API (model: {settings.llm.gemini_model})")
        elif settings.llm.gemini_via_aipipe and settings.llm.aipipe_token:
            self._gemini_configured = True
            logger.warning("Gemini configured via aipipe proxy (fallback only)")
        else:
            logger.warning("No GEMINI_API_KEY found - Gemini not available")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
    
    def _select_model(self) -> str:
        """Select which model to use based on availability.
        
        Priority:
        1. Aipipe (Primary - as requested by user)
        2. Gemini (Fallback)
        """
        
        # Prefer Aipipe as primary
        if self._aipipe_configured:
            logger.debug("Using aipipe (primary)")
            return "aipipe"
            
        if self._gemini_configured:
            logger.debug("Using Gemini (fallback)")
            return "gemini"
        
        raise RuntimeError("No LLM client available")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        json_response: bool = False,
        timeout: int = 30  # FIX #4: Add timeout parameter
    ) -> str:
        """Generate a response from the LLM with retry logic and timeout.
        
        FIX #4: Enforces timeout to prevent hanging LLM calls.
        """
        import asyncio
        
        model_choice = self._select_model()
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                # FIX #4: Wrap LLM call in asyncio.wait_for() to enforce timeout
                if model_choice == "aipipe":
                    response = await asyncio.wait_for(
                        self._generate_aipipe(prompt, max_tokens, temperature, json_response),
                        timeout=timeout
                    )
                else:
                    response = await asyncio.wait_for(
                        self._generate_gemini(prompt, max_tokens, temperature),
                        timeout=timeout
                    )
                return response
            except asyncio.TimeoutError:
                logger.warning(f"LLM call timed out after {timeout}s (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(f"LLM call timed out after {max_retries} attempts")
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limiting with retry
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 seconds
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Primary model ({model_choice}) failed: {repr(e)}")
                
                # Try fallback on final failure
                if model_choice == "aipipe" and self._gemini_configured:
                    logger.info("Falling back to Gemini")
                    try:
                        return await self._generate_gemini(prompt, max_tokens, temperature)
                    except Exception as fallback_e:
                        logger.error(f"Fallback to Gemini also failed: {repr(fallback_e)}")
                        raise
                elif model_choice == "gemini" and self._aipipe_configured:
                    logger.info("Falling back to aipipe")
                    try:
                        return await self._generate_aipipe(prompt, max_tokens, temperature, json_response)
                    except Exception as fallback_e:
                        logger.error(f"Fallback to aipipe also failed: {repr(fallback_e)}")
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
        """Generate using aipipe via OpenAI-compatible endpoint.
        
        Uses: https://aipipe.org/openai/v1/chat/completions (default)
        Models: gpt-5-nano, gpt-4o-mini, etc.
        """
        
        if not self._aipipe_configured or not self._http_client:
            raise RuntimeError("Aipipe client not initialized")
        
        # Use configured base URL
        base_url = settings.llm.aipipe_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        
        model = settings.llm.aipipe_model
        
        # If using OpenRouter endpoint, ensure prefix
        if "openrouter" in base_url and "/" not in model:
            model = f"openai/{model}"
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        # Handle max_tokens vs max_completion_tokens for newer models (gpt-5, o1, etc.)
        # gpt-5-nano requires max_completion_tokens and fixed temperature
        if "gpt-5" in model or "o1-" in model:
            # Reasoning models consume tokens for internal thought processes before generating output.
            # If the limit is too low (e.g. 800), they may use it all on reasoning and return empty content.
            # We enforce a higher minimum to prevent this.
            payload["max_completion_tokens"] = max(max_tokens, 60000)
            # These models often require temperature=1 or don't support it
            payload["temperature"] = 1.0
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        
        headers = {
            "Authorization": f"Bearer {settings.llm.aipipe_token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sanand0/tools-in-data-science-public",
            "X-Title": "TDS Quiz Solver"
        }
        
        logger.debug(f"Calling aipipe: {url} with model {model}")
        response = await self._http_client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Aipipe failed with status {response.status_code}: {response.text}")
            response.raise_for_status()
        
        data = response.json()
        
        # Track token usage
        if "usage" in data:
            self.token_tracker.aipipe_used += data["usage"].get("total_tokens", 0)
        
        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Aipipe returned no choices: {data}")
            
        content = data["choices"][0]["message"]["content"]
        
        if not content or not content.strip():
            finish_reason = data["choices"][0].get("finish_reason")
            logger.error(f"Aipipe returned empty content. Finish reason: {finish_reason}. Full response: {json.dumps(data)}")
            raise ValueError(f"Aipipe returned empty response (finish_reason: {finish_reason})")
            
        return content
        
        logger.debug(f"Aipipe response: {content[:100]}...")
        
        return content
    
    async def _generate_gemini(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using Gemini - ALWAYS prefer direct API.
        
        Priority:
        1. Direct Google API (if GEMINI_API_KEY is set)
        2. Aipipe proxy (fallback only)
        """
        
        if not self._gemini_configured or not self._http_client:
            raise RuntimeError("Gemini client not initialized")
        
        # ALWAYS prefer direct API if key is available
        if settings.llm.gemini_api_key:
            logger.debug("Using direct Gemini API")
            return await self._generate_gemini_direct(prompt, max_tokens, temperature)
        elif settings.llm.gemini_via_aipipe and settings.llm.aipipe_token:
            # Fallback to aipipe proxy only if no direct key
            logger.debug("Using Gemini via aipipe proxy (fallback)")
            return await self._generate_gemini_via_aipipe(prompt, max_tokens, temperature)
        else:
            raise RuntimeError("No Gemini API configuration available")
    
    async def _generate_gemini_via_aipipe(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using Gemini via aipipe's native Gemini endpoint.
        
        Uses: https://aipipe.org/geminiv1beta/models/MODEL:generateContent
        Auth: Authorization: Bearer {AIPIPE_TOKEN}
        
        Handles both standard models (2.0) and thinking models (2.5+).
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
        
        # Gemini 2.5+ are "thinking models" - disable thinking for fast responses
        is_thinking_model = "2.5" in model or "gemini-3" in model
        
        # Gemini native request format
        generation_config: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
        
        payload: dict = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": generation_config
        }
        
        # Disable thinking for 2.5 models
        if is_thinking_model:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": 0
            }
            logger.debug(f"Thinking model ({model}): disabled thinking for fast response")
        
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
            if not content or not content.strip():
                raise ValueError("Gemini returned empty text in response")
            
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
        """Generate using direct Gemini API.
        
        Handles both standard models (2.0) and thinking models (2.5+).
        For 2.5+ models, we DISABLE thinking to get fast, direct responses.
        """
        
        if not self._http_client or not settings.llm.gemini_api_key:
            raise RuntimeError("Gemini direct API not configured")
        
        # Use Gemini's generateContent API
        model = settings.llm.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        # Gemini 2.5+ are "thinking models" - they can use up to 24K tokens just for thinking!
        # For our quiz solver, we want fast responses, so DISABLE thinking with thinkingBudget=0
        is_thinking_model = "2.5" in model or "gemini-3" in model
        
        generation_config: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
        
        # Build payload
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config
        }
        
        # Disable thinking for 2.5 models to get fast responses
        if is_thinking_model:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": 0  # Disable thinking - we want fast, direct answers
            }
            logger.debug(f"Thinking model ({model}): disabled thinking for fast response")
        
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
        
        # Parse response with robust error handling
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                error_msg = data.get("error", {}).get("message", "No candidates in response")
                logger.error(f"Gemini direct API error: {error_msg}, full response: {data}")
                raise ValueError(f"Gemini API error: {error_msg}")
            
            content_obj = candidates[0].get("content", {})
            parts = content_obj.get("parts", [])
            if not parts:
                finish_reason = candidates[0].get("finishReason", "")
                if finish_reason == "SAFETY":
                    raise ValueError("Response blocked by safety filters")
                logger.error(f"Gemini direct: No parts in response, full data: {data}")
                raise ValueError("No parts in response content")
            
            content = parts[0].get("text", "")
            if not content or not content.strip():
                raise ValueError("Gemini direct returned empty text in response")
            
            logger.debug(f"Gemini (direct) response: {content[:100]}...")
            return content
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse Gemini direct response: {e}, data: {data}")
            raise ValueError(f"Invalid Gemini response structure: {e}")
    
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
    
    async def classify_task_dynamically(self, question: str, context: dict) -> dict:
        """
        Use LLM to classify the task type and requirements.
        This replaces brittle keyword matching with intelligent analysis.
        
        Args:
            question: The raw question text
            context: Available context (files, data, configs, etc.)
            
        Returns:
            Classification dict with task_type, answer_format, personalization, etc.
        """
        
        context_summary = []
        if context.get('dominant_color'):
            context_summary.append("- Image data available with dominant_color")
        if context.get('github_config'):
            context_summary.append("- GitHub API config available")
        if context.get('audio_transcript'):
            context_summary.append("- Audio transcript available")
        if context.get('dataframe') is not None:
            context_summary.append("- DataFrame available with data to analyze")
        if context.get('logs_data'):
            context_summary.append("- Logs data available")
        
        context_str = "\n".join(context_summary) if context_summary else "- No special data sources"
        
        classify_prompt = f"""Analyze this quiz question and classify the task type.

QUESTION: {question}

AVAILABLE CONTEXT:
{context_str}

Return ONLY valid JSON with this structure:
{{
    "task_type": "image_analysis|api_call|data_analysis|command_generation|audio_transcription|text_extraction|csv_to_json|chart_selection|other",
    "answer_format": "hex_color|integer|float|json|command_string|text_phrase|csv_json|boolean|other",
    "has_personalization": true/false,
    "personalization_type": "email_length_mod_2|email_length_mod_3|email_length_mod_5|email_checksum|none",
    "requires_api": true/false,
    "api_type": "github|custom|none",
    "requires_data_transformation": true/false,
    "transformations": ["snake_case", "iso_dates", "integer_values", "sorted_by_id"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of classification"
}}

Classification logic:
- If question mentions image/color/rgb/hex → image_analysis
- If question mentions API/GitHub/repos/trees → api_call  
- If question mentions DataFrame/CSV/JSON transformation → data_analysis or csv_to_json
- If question mentions shell/command/git/uv → command_generation
- If question mentions audio/transcript/spoken → audio_transcription
- If question mentions email length/offset/mod → has_personalization=true
- Look for personalization patterns: "length of your email", "email mod", "offset"

Answer:"""
        
        try:
            # FIX #4: Use timeout=40 for classification calls
            response = await self.generate(classify_prompt, max_tokens=600, temperature=0.1, timeout=40)
            response = response.strip()
            
            # Clean JSON markers
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            classification = json.loads(response.strip())
            logger.info(f"✓ Task classified: {classification.get('task_type')} (confidence: {classification.get('confidence')})")
            return classification
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse classification JSON: {e}")
            return {
                "task_type": "other",
                "answer_format": "other",
                "has_personalization": False,
                "confidence": 0.0
            }
    
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
