"""
Generic task handlers for different question types.

This module replaces hardcoded if/elif chains with dynamic, reusable handlers.
Each handler can process ANY question of its type, regardless of specific phrasing.
"""

import json
import httpx
import pandas as pd
from typing import Optional, Any, Dict
from .logging_utils import logger
from .llm_client import llm_client


async def handle_image_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[str]:
    """
    Generic image analysis handler.
    Works for: color extraction, image comparison, rotation detection, etc.
    """
    logger.info("🖼️  Handling image analysis task")
    
    # Extract dominant color if available
    if 'dominant_color' in context:
        color_value = context['dominant_color']
        format_needed = classification.get("answer_format", "hex_color")
        
        if format_needed == "hex_color":
            # Ensure proper hex format
            if not color_value.startswith('#'):
                color_value = f"#{color_value}"
            return color_value[:7]  # #rrggbb
        
        elif format_needed == "rgb":
            # Convert hex to RGB tuple
            hex_val = color_value.lstrip('#')
            r, g, b = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
            return f"rgb({r},{g},{b})"
    
    # For image comparison/diff
    if 'pixel_diff' in context:
        return str(context['pixel_diff'])
    
    # Fallback: ask LLM to analyze image data
    if any(key.startswith('image') or 'color' in key for key in context.keys()):
        image_info = "\n".join([f"{k}: {v}" for k, v in context.items() 
                                if isinstance(v, (str, int, float))])
        
        image_prompt = f"""Answer this image-related question using the provided image data.

QUESTION: {question}

IMAGE DATA:
{image_info}

Return ONLY the answer value (e.g., #rrggbb for color)."""
        
        answer = await llm_client.generate(image_prompt, max_tokens=100, temperature=0.1)
        return answer.strip()
    
    return None


async def handle_api_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[Any]:
    """
    Generic API call handler.
    Works for: GitHub API, custom APIs, REST endpoints, etc.
    """
    logger.info("🌐 Handling API call task")
    
    api_type = classification.get("api_type", "custom")
    
    # GitHub API specialization
    if api_type == "github" and context.get('github_config'):
        config = context['github_config']
        
        # Construct GitHub API URL
        owner = config.get('owner', '')
        repo = config.get('repo', '')
        sha = config.get('sha', '')
        
        if owner and repo and sha:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
            
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(api_url)
                    resp.raise_for_status()
                    data = resp.json()
                
                # Extract tree data
                tree = data.get('tree', [])
                
                # Filter by path prefix and extension
                path_prefix = config.get('pathPrefix', '')
                extension = config.get('extension', '.md')
                
                count = sum(1 for item in tree 
                           if item.get('type') == 'blob' and
                              item.get('path', '').startswith(path_prefix) and
                              item.get('path', '').endswith(extension))
                
                logger.info(f"   ✓ GitHub API: Found {count} {extension} files under {path_prefix}")
                
                # Apply personalization if needed
                if classification.get("has_personalization"):
                    offset = calculate_personalization_offset(
                        session.email,
                        classification.get("personalization_type")
                    )
                    logger.info(f"   ✓ Applying personalization offset: {offset}")
                    count += offset
                
                return count
                
            except Exception as e:
                logger.error(f"   ❌ GitHub API call failed: {e}")
                return None
    
    # Generic API call using LLM guidance
    api_prompt = f"""Parse this question to extract API call details.

QUESTION: {question}

Return ONLY valid JSON:
{{
    "method": "GET|POST",
    "url": "<full API URL>",
    "headers": {{}},
    "params": {{}}
}}"""
    
    try:
        response = await llm_client.generate(api_prompt, max_tokens=300, temperature=0.1)
        response = response.strip()
        
        # Clean JSON markers
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        api_spec = json.loads(response)
        
        # Execute API call
        async with httpx.AsyncClient(timeout=30) as client:
            if api_spec.get('method', 'GET') == 'GET':
                resp = await client.get(
                    api_spec['url'],
                    headers=api_spec.get('headers', {})
                )
            else:
                resp = await client.post(
                    api_spec['url'],
                    headers=api_spec.get('headers', {}),
                    json=api_spec.get('params', {})
                )
            
            resp.raise_for_status()
            api_data = resp.json()
        
        # Use LLM to extract answer from response
        extract_prompt = f"""Extract the answer from this API response.

QUESTION: {question}
API RESPONSE (truncated): {json.dumps(api_data)[:2000]}

Return ONLY the final answer value."""
        
        answer = await llm_client.generate(extract_prompt, max_tokens=200, temperature=0.1)
        return answer.strip()
        
    except Exception as e:
        logger.error(f"   ❌ API call failed: {e}")
        return None


async def handle_data_analysis_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[Any]:
    """
    Generic data analysis handler.
    Works for: aggregations, filtering, transformations, statistical analysis, etc.
    """
    logger.info("📊 Handling data analysis task")
    
    df = context.get('dataframe')
    
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.warning("   ⚠️  No DataFrame available for analysis")
        return None
    
    # Generate analysis code using LLM
    from .data_sourcing import get_dataframe_info
    df_info = get_dataframe_info(df)
    
    analysis_prompt = f"""Generate Python pandas code to answer this question.

QUESTION: {question}

DATAFRAME INFO:
- Shape: {df_info.get('shape')}
- Columns: {df_info.get('columns')}
- Dtypes: {df_info.get('dtypes')}
- Sample:
{df_info.get('sample', '')}

Generate ONLY Python code (no imports, df is already loaded):
- Store final answer in variable 'answer'
- Handle all requirements from question
- For numeric answers, ensure correct data type (int vs float)

Code:"""
    
    try:
        code = await llm_client.generate(analysis_prompt, max_tokens=500, temperature=0.1)
        code = code.strip()
        
        # Clean code markers
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        # Execute code
        from .analysis import execute_analysis_code
        answer, output, error = await execute_analysis_code(code.strip(), df)
        
        if error:
            logger.error(f"   ❌ Analysis code error: {error}")
            return None
        
        # Apply personalization if needed
        if classification.get("has_personalization") and isinstance(answer, (int, float)):
            offset = calculate_personalization_offset(
                session.email,
                classification.get("personalization_type")
            )
            logger.info(f"   ✓ Applying personalization offset: {offset}")
            answer = int(answer) + offset
        
        logger.info(f"   ✓ Analysis result: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Data analysis failed: {e}")
        return None


async def handle_csv_to_json_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[str]:
    """
    Generic CSV to JSON transformation handler.
    Works for: normalization, renaming, type conversion, sorting, etc.
    """
    logger.info("🔄 Handling CSV to JSON transformation")
    
    df = context.get('dataframe')
    
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    
    # Get transformation requirements from classification
    transformations = classification.get("transformations", [])
    
    # Generate transformation code using LLM
    from .data_sourcing import get_dataframe_info
    df_info = get_dataframe_info(df)
    
    transform_prompt = f"""Generate Python pandas code to transform this DataFrame to JSON.

QUESTION: {question}

DATAFRAME:
- Columns: {df_info.get('columns')}
- Sample:
{df_info.get('sample', '')}

REQUIREMENTS: {transformations}

Generate ONLY Python code that:
1. Applies all transformations
2. Converts to JSON array format
3. Stores result in 'answer' variable

Code:"""
    
    try:
        code = await llm_client.generate(transform_prompt, max_tokens=500, temperature=0.1)
        code = code.strip()
        
        # Clean markers
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        # Execute transformation
        from .analysis import execute_analysis_code
        answer, output, error = await execute_analysis_code(code.strip(), df)
        
        if error:
            logger.error(f"   ❌ Transformation error: {error}")
            # Fallback: basic transformation
            answer = df.to_json(orient='records')
        
        logger.info(f"   ✓ JSON transformation complete")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ CSV to JSON failed: {e}")
        return None


async def handle_command_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[str]:
    """
    Generic command generation handler.
    Works for: git commands, shell commands, uv commands, etc.
    
    OPTIMIZED: Single LLM call, no duplicate classification.
    """
    logger.info("⌨️  Handling command generation task")
    
    # FIX #1 & #2: Use optimized single-call command generation
    # Don't re-classify (already done in pipeline)
    try:
        command = await llm_client.generate_complete_command(question, session)
        
        if not command:
            logger.warning("   ⚠️  Empty command generated")
            return None
        
        logger.info(f"   ✓ Command ready: {command[:100]}")
        return command
        
    except Exception as e:
        logger.error(f"   ❌ Command generation failed: {e}")
        return None


# OLD IMPLEMENTATION REMOVED - was doing duplicate work:
# - Re-classification (already done)
# - Separate URL extraction
# - Separate command building  
# - Separate validation
# Now: Single optimized call


async def _handle_command_task_old_slow(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[str]:
    """
    OLD SLOW VERSION - DEPRECATED
    Kept for reference only. Do not use.
    """
    logger.info("⌨️  Handling command generation task")
    
    # Let LLM generate the ENTIRE command (don't extract and wrap)
    command_prompt = f"""Generate the exact shell command(s) requested.

QUESTION: {question}

Instructions:
- Generate the COMPLETE command string
- Do NOT add quotes around URLs unless they contain spaces
- Preserve exact format requested (newlines, separators, etc.)
- For git commands, ensure proper quoting of messages
- Return ONLY the command(s), nothing else

Command:"""
    
    try:
        command = await llm_client.generate(command_prompt, max_tokens=300, temperature=0.1)
        command = command.strip()
        
        # Validate command
        validation_prompt = f"""Does this command correctly answer the question?

QUESTION: {question}
COMMAND: {command}

Return ONLY: "yes" or "no" with brief reason if no."""
        
        validation = await llm_client.generate(validation_prompt, max_tokens=50, temperature=0.1)
        
        if "no" in validation.lower():
            logger.warning(f"   ⚠️  Command validation failed: {validation}")
            # Regenerate
            command = await llm_client.generate(command_prompt, max_tokens=300, temperature=0.1)
            command = command.strip()
        
        logger.info(f"   ✓ Generated command: {command}")
        return command
        
    except Exception as e:
        logger.error(f"   ❌ Command generation failed: {e}")
        return None


async def handle_audio_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[str]:
    """
    Generic audio transcription handler.
    Works for: transcription, phrase extraction, spoken content analysis, etc.
    """
    logger.info("🎵 Handling audio transcription task")
    
    transcript = context.get('audio_transcript', '')
    
    if not transcript:
        logger.warning("   ⚠️  No audio transcript available")
        return None
    
    # Process transcript according to question requirements
    process_prompt = f"""Process this audio transcript according to the question.

QUESTION: {question}
TRANSCRIPT: {transcript}

Instructions:
- Extract exactly what the question asks for
- Preserve format (lowercase, spaces, digits, etc.)
- Return ONLY the answer

Answer:"""
    
    try:
        answer = await llm_client.generate(process_prompt, max_tokens=200, temperature=0.1)
        answer = answer.strip()
        
        logger.info(f"   ✓ Processed transcript answer: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Audio processing failed: {e}")
        return transcript.lower().strip()  # Fallback


async def handle_text_extraction_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[str]:
    """
    Generic text extraction handler.
    Works for: path extraction, URL extraction, specific text finding, etc.
    """
    logger.info("📝 Handling text extraction task")
    
    # Use LLM to extract from question itself
    extract_prompt = f"""Extract the exact text/path/URL requested.

QUESTION: {question}

Return ONLY the extracted value, nothing else."""
    
    try:
        answer = await llm_client.generate(extract_prompt, max_tokens=100, temperature=0.1)
        answer = answer.strip()
        
        logger.info(f"   ✓ Extracted: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Text extraction failed: {e}")
        return None


async def handle_fallback_task(
    question: str,
    context: dict,
    classification: dict,
    session: Any
) -> Optional[str]:
    """
    Fallback handler for unclassified or unknown task types.
    Uses LLM to answer directly from context.
    """
    logger.info("🔮 Handling via fallback (LLM direct answer)")
    
    # Build context summary
    context_summary = []
    for key, value in context.items():
        if isinstance(value, (str, int, float, bool)):
            context_summary.append(f"{key}: {value}")
        elif isinstance(value, dict):
            context_summary.append(f"{key}: {json.dumps(value)[:200]}")
        elif isinstance(value, pd.DataFrame):
            context_summary.append(f"{key}: DataFrame with shape {value.shape}")
    
    context_str = "\n".join(context_summary[:20])  # Limit to 20 items
    
    answer = await llm_client.answer_from_context(context_str, question)
    logger.info(f"   ✓ Fallback answer: {answer}")
    return answer


def calculate_personalization_offset(email: str, personalization_type: str) -> int:
    """
    Calculate personalization offset based on email and type.
    
    Args:
        email: User's email address
        personalization_type: Type of personalization (e.g., "email_length_mod_2")
        
    Returns:
        Integer offset value
    """
    if not personalization_type or personalization_type == "none":
        return 0
    
    email_len = len(email)
    
    if "mod_2" in personalization_type or "mod 2" in personalization_type:
        return email_len % 2
    elif "mod_3" in personalization_type or "mod 3" in personalization_type:
        return email_len % 3
    elif "mod_5" in personalization_type or "mod 5" in personalization_type:
        return email_len % 5
    elif "checksum" in personalization_type:
        return sum(ord(c) for c in email) % 100
    
    # Default: try to parse from string
    import re
    match = re.search(r'mod[_ ](\d+)', personalization_type)
    if match:
        modulo = int(match.group(1))
        return email_len % modulo
    
    return 0


# Handler routing map
TASK_HANDLERS = {
    "image_analysis": handle_image_task,
    "api_call": handle_api_task,
    "data_analysis": handle_data_analysis_task,
    "csv_to_json": handle_csv_to_json_task,
    "command_generation": handle_command_task,
    "audio_transcription": handle_audio_task,
    "text_extraction": handle_text_extraction_task,
    "chart_selection": handle_fallback_task,  # Charts are handled by LLM
    "other": handle_fallback_task
}
