"""
LLM-driven analysis module.

This module contains all LLM-driven logic for solving quiz questions.
The LLM analyzes each question dynamically - no hardcoded pattern matching.

=== APPROACH ===

1. Question comes in → LLM analyzes it to understand WHAT is being asked
2. LLM determines → Data to fetch, processing steps, answer format
3. Execute dynamically → Generate code, run it, return result

=== GA MODULES COVERED ===

- GA1: Development Tools (uv, git, bash)
- GA2: Deployment (Docker, Vercel, GitHub Actions)  
- GA3: AI Coding (code generation, prompts)
- GA4: LLMs (API calls, embeddings, function calling)
- GA5: Data Sourcing (wget, curl, APIs, scraping)
- GA6: Data Preparation (pandas, SQL, DuckDB, cleaning)
- GA7: Data Analysis (statistics, geospatial, networks)
- GA8: Data Visualization (charts, base64 PNG)

=== KEY PRINCIPLES ===

1. PARAMETERIZED: Questions use email-based seeds for personalization
2. EXACT ANSWERS: Numbers must be precise, formats must match
3. TIME PRESSURE: 3 minutes total - optimize for speed
4. LLM-FIRST: Let the LLM decide, don't over-engineer patterns
"""

import json
import re
import math
from typing import Any, Optional, List, Dict
import pandas as pd

from .logging_utils import logger


# TDS Course Question Patterns - FLEXIBLE detection (not exhaustive)
# These are HINTS for the LLM, not strict matching rules
TDS_QUESTION_HINTS = {
    # Pattern groups that suggest certain task types
    "data_analysis_hints": ["sum", "total", "count", "filter", "where", "average", "group by", "rows"],
    "sql_hints": ["sql", "select", "from", "duckdb", "sqlite", "query"],
    "api_hints": ["api", "endpoint", "github", "repository", "request", "fetch"],
    "command_hints": ["git", "uv", "curl", "wget", "bash", "run", "command"],
    "chart_hints": ["chart", "plot", "visualization", "base64", "png", "image"],
    "geo_hints": ["latitude", "longitude", "distance", "km", "miles", "haversine"],
    "network_hints": ["graph", "node", "edge", "shortest path", "networkx"],
}

# Answer format detection - used as fallback hints
ANSWER_FORMATS = {
    "number": ["sum", "count", "total", "average", "how many", "percentage", "ratio"],
    "hash": ["SHA", "hash", "checksum", "MD5"],
    "command": ["command", "run", "execute", "bash", "terminal"],
    "json": ["JSON", "array", "object", "{}"],
    "base64_image": ["base64", "PNG", "image", "chart", "plot", "encode"],
    "hex_color": ["color", "hex", "#", "RGB"],
    "string": ["name", "text", "value", "extract"],
}


async def analyze_question_deeply(llm_client, question_text: str, context: dict) -> dict:
    """
    Use LLM to deeply analyze the question and determine the solution strategy.
    
    This is the BRAIN of the quiz solver - it must understand the SOUL of TDS questions:
    
    1. PARAMETERIZED: Same logic, different values per student (email-based seed)
    2. EXACT ANSWERS: Numbers must be precise, hashes must match exactly
    3. TIME PRESSURE: 3 minutes total, optimize for speed
    4. PRACTICAL: Tests ability to DO things, not theory
    
    Task types mapping to TDS modules:
    - data_analysis: GA6/GA7 - pandas, SQL, aggregation, statistics
    - web_scrape: GA5 - CSS selectors, BeautifulSoup, HTML tables
    - command_generation: GA1/GA2 - git, uv, curl, wget, bash commands
    - api_call: GA4/GA5 - GitHub API, REST APIs, LLM endpoints
    - text_extraction: GA5/GA6 - PDF tables, page-specific data
    - audio_transcription: GA4 - Gemini audio API, Whisper
    - image_analysis: GA4/GA8 - dominant color, OCR, base64
    - network_analysis: GA7 - NetworkX, shortest paths, degree
    - geospatial: GA7 - Haversine distance, lat/long calculations
    - llm_task: GA4 - embeddings, function calling, prompt injection
    - json_transform: GA6 - format conversion, structured output
    - visualization: GA8 - matplotlib charts, base64 PNG encoding
    - intro_page: Start/welcome pages
    """
    
    # Build context summary for LLM
    context_summary = []
    if context.get('audio_transcript'):
        context_summary.append(f"Audio transcript available: {context['audio_transcript'][:200]}...")
    if context.get('dominant_color'):
        context_summary.append(f"Image analyzed - dominant color: {context['dominant_color']}")
    if context.get('pdf_text'):
        context_summary.append(f"PDF text available: {len(context['pdf_text'])} chars")
    if context.get('pdf_tables'):
        context_summary.append(f"PDF tables: {len(context['pdf_tables'])} tables extracted")
    if context.get('dataframe') is not None:
        df = context['dataframe']
        context_summary.append(f"DataFrame available: {df.shape}, columns: {list(df.columns)}")
    if context.get('github_config'):
        context_summary.append(f"GitHub config: {context['github_config']}")
    if context.get('logs_data'):
        context_summary.append(f"Logs data: {len(context['logs_data'])} entries")
    if context.get('webpage_text'):
        context_summary.append(f"Webpage text: {len(context['webpage_text'])} chars")
    if context.get('html_content'):
        context_summary.append(f"HTML content: {len(context['html_content'])} chars")
    if context.get('edges_data') or context.get('graph_data'):
        context_summary.append(f"Graph/network data available")
    if context.get('zip_files'):
        context_summary.append(f"ZIP files: {len(context['zip_files'])} files")
    
    context_str = "\n".join(context_summary) if context_summary else "No additional context available"
    
    # Enhanced analysis prompt based on TDS course patterns
    analysis_prompt = f"""You are an expert at solving TDS (Tools in Data Science) quiz questions.

QUESTION:
{question_text}

AVAILABLE CONTEXT:
{context_str}

TDS QUESTION PATTERNS TO RECOGNIZE:

1. DATA AGGREGATION: "sum", "total", "count", "average", "how many" → compute numeric answer
2. FILTERING: "where", "filter", "only", "rows with" → filter then aggregate
3. GITHUB/API: "repository", "files", "API endpoint" → make API call, count results
4. COMMAND: "git", "uv", "curl", "wget", "bash" → generate shell command
5. PDF/TABLE: "page X", "table", "column" → extract from specific location
6. CHART: "plot", "chart", "visualization", "base64" → generate matplotlib, encode PNG
7. DISTANCE: "latitude", "longitude", "distance", "km" → Haversine formula
8. NETWORK: "shortest path", "node", "edge", "degree" → NetworkX operations
9. EMBEDDING: "most similar", "embedding", "cosine" → compute similarity
10. INTRO PAGE: "start", "click", "begin", "welcome" → return "start"

RULES:
- Use URLs EXACTLY as they appear in the question or context. Do NOT invent or hallucinate URLs.
- If a command is requested, use the exact parameters specified.
- For command generation tasks, ensure all URLs, filenames, and parameters match the question text exactly.

Return ONLY valid JSON:
{{
    "task_type": "data_analysis|web_scrape|command_generation|api_call|text_extraction|audio_transcription|image_analysis|network_analysis|geospatial|llm_task|json_transform|visualization|intro_page",
    "answer_format": "number|string|json|boolean|command|hash|hex_color|base64_image",
    "solution_strategy": "<EXACT steps: 1. Load data from X, 2. Filter by Y, 3. Calculate Z>",
    "column_to_aggregate": "<if data question, which column to sum/count>",
    "filter_condition": "<if filtering needed, what condition>",
    "page_number": <if PDF, which page (0-indexed)>,
    "api_endpoint": "<if API call, construct the URL>",
    "personalization": {{
        "uses_email": true/false,
        "offset_formula": "<e.g., len(email) % 10>"
    }},
    "precision": "<decimal places if numeric>",
    "confidence": 0.0-1.0,
    "fallback_answer": "<safe default: 'start' for intro, 0 for numbers>"
}}

CRITICAL: Be SPECIFIC. "Sum column X where Y > Z" not "analyze the data"."""

    try:
        response = await llm_client.generate(analysis_prompt, max_tokens=800, temperature=0.1)
        
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```json"] else lines[1:])
        
        analysis = json.loads(response)
        logger.info(f"   📊 Question analysis: task={analysis.get('task_type')}, format={analysis.get('answer_format')}")
        logger.info(f"   📊 Strategy: {analysis.get('solution_strategy', 'N/A')[:80]}...")
        return analysis
        
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"   ⚠️ Question analysis failed: {e}")
        # Return default analysis
        return {
            "task_type": "llm_task",
            "answer_format": "string",
            "solution_strategy": "Use LLM to answer directly from context",
            "confidence": 0.3,
            "fallback_answer": "start"
        }


async def llm_driven_data_analysis(
    llm_client,
    df: pd.DataFrame,
    context: dict,
    question: str,
    analysis: dict,
    session: Any
) -> Any:
    """
    Use LLM to generate and execute analysis code on a DataFrame.
    No hardcoded logic - LLM decides what to compute.
    """
    from .data_sourcing import get_dataframe_info
    from .analysis import execute_analysis_code
    
    df_info = get_dataframe_info(df)
    
    # Include solution strategy from analysis
    guidance = analysis.get('solution_strategy', '')
    transformations = analysis.get('transformations', [])
    personalization = analysis.get('personalization', {})
    
    code_prompt = f"""Generate Python pandas code to answer this question.

QUESTION: {question}

CRITICAL RULES:
1. Variable 'df' already contains the loaded DataFrame - USE IT DIRECTLY
2. DO NOT create dummy data or sample data
3. DO NOT use pd.read_csv(), pd.read_json(), or any file loading
4. DO NOT redefine 'df' - it already has the real data

ACTUAL DATA IN 'df':
- Shape: {df_info.get('shape')}
- Columns: {df_info.get('columns')}
- First 5 rows:
{df_info.get('sample', '')}

SOLUTION GUIDANCE: {guidance}

{"PERSONALIZATION: Email length is " + str(len(session.email)) + ", use for offset calculation: " + personalization.get('offset_calculation', '') if personalization.get('uses_email') else ""}

Generate ONLY the analysis code:
- Work with existing 'df' variable (contains real data above)
- Store final answer in 'answer' variable
- No imports needed (pandas and numpy available as pd/np)

Code:"""

    try:
        code = await llm_client.generate(code_prompt, max_tokens=600, temperature=0.1)
        
        # Clean code
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```python"] else lines[1:])
        
        # Remove any markdown
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        
        # IMPORTANT: Strip any lines that try to redefine 'df' or load data
        cleaned_lines = []
        skip_block = False
        for line in code.split('\n'):
            line_lower = line.lower().strip()
            # Skip lines that redefine df or load files
            if any(pattern in line_lower for pattern in [
                'pd.read_csv', 'pd.read_json', 'pd.read_excel', 
                'pd.dataframe(', "df = pd.", "df=pd.",
                "# for demonstration", "# assuming", "# let's create",
                "# create a dummy", "# sample data"
            ]):
                skip_block = True
                continue
            # Skip multi-line dictionary definitions for dummy data
            if skip_block:
                if line.strip().startswith('}') or line.strip() == '':
                    skip_block = False
                continue
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines)
        
        logger.info(f"   🤖 Generated code:")
        for line in code.split('\n')[:8]:
            logger.info(f"      {line}")
        
        # Execute the code
        answer, output, error = await execute_analysis_code(code, df)
        
        if error:
            logger.warning(f"   ⚠️ Code error: {error}")
            # Ask LLM to fix the code
            fix_prompt = f"""The code failed with error: {error}

Original code:
{code}

Fix the code and return ONLY the corrected Python code:"""
            
            fixed_code = await llm_client.generate(fix_prompt, max_tokens=600, temperature=0.1)
            fixed_code = fixed_code.strip()
            if fixed_code.startswith("```"):
                lines = fixed_code.split("\n")
                fixed_code = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```python"] else lines[1:])
            
            answer, output, error = await execute_analysis_code(fixed_code, df)
            
            if error:
                logger.warning(f"   ⚠️ Fixed code also failed: {error}")
                return None
        
        logger.info(f"   ✓ Analysis result: {str(answer)[:100]}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Data analysis failed: {e}")
        return None


async def llm_driven_command_generation(
    llm_client,
    question: str,
    analysis: dict,
    session: Any
) -> Optional[str]:
    """
    Use LLM to generate shell/git/uv commands.
    No hardcoded command patterns.
    """
    
    command_prompt = f"""Generate the exact command string for this task.

QUESTION: {question}

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}

{"USER EMAIL: " + session.email if analysis.get('personalization', {}).get('uses_email') else ""}

Return ONLY the command string (no explanations, no code blocks).
If multiple commands are needed, separate them with newlines.
Include all required flags and arguments exactly as specified."""

    try:
        command = await llm_client.generate(command_prompt, max_tokens=300, temperature=0.1)
        command = command.strip()
        
        # Remove any markdown code blocks
        if command.startswith("```"):
            lines = command.split("\n")
            command = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```bash", "```shell"] else lines[1:])
        
        command = command.strip('`').strip()
        
        logger.info(f"   ✓ Generated command: {command[:100]}")
        return command
        
    except Exception as e:
        logger.error(f"   ❌ Command generation failed: {e}")
        return None


async def llm_driven_api_call(
    llm_client,
    question: str,
    analysis: dict,
    context: dict,
    session: Any
) -> Any:
    """
    Use LLM to plan and execute API calls (like GitHub API).
    """
    import httpx
    
    # Check if we have API config in context
    api_config = context.get('github_config') or {}
    
    api_prompt = f"""Plan the API call needed to answer this question.

QUESTION: {question}

AVAILABLE CONFIG: {json.dumps(api_config) if api_config else "None - extract from question"}

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}

Return ONLY valid JSON:
{{
    "api_type": "github|custom|none",
    "method": "GET|POST",
    "url": "<full API URL>",
    "headers": {{}},
    "params": {{}},
    "extract_from_response": "<what to extract, e.g., 'count items where path ends with .md'>"
}}"""

    try:
        response = await llm_client.generate(api_prompt, max_tokens=400, temperature=0.1)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```json"] else lines[1:])
        
        api_plan = json.loads(response)
        
        if api_plan.get('api_type') == 'none':
            return None
        
        # Execute the API call
        async with httpx.AsyncClient(timeout=30) as client:
            if api_plan.get('method', 'GET') == 'GET':
                resp = await client.get(
                    api_plan['url'],
                    headers=api_plan.get('headers', {}),
                    params=api_plan.get('params', {})
                )
            else:
                resp = await client.post(
                    api_plan['url'],
                    headers=api_plan.get('headers', {}),
                    json=api_plan.get('params', {})
                )
            
            resp.raise_for_status()
            data = resp.json()
        
        # Use LLM to extract answer from response
        extract_prompt = f"""Extract the answer from this API response.

QUESTION: {question}
EXTRACTION GOAL: {api_plan.get('extract_from_response', 'Extract the relevant value')}

API RESPONSE (first 3000 chars):
{json.dumps(data)[:3000]}

{"PERSONALIZATION: Email length is " + str(len(session.email)) + ", offset formula: " + analysis.get('personalization', {}).get('offset_calculation', '') if analysis.get('personalization', {}).get('uses_email') else ""}

Return ONLY the final answer value (number, string, etc.)."""

        answer = await llm_client.generate(extract_prompt, max_tokens=200, temperature=0.1)
        answer = answer.strip()
        
        # Try to convert to number if it looks like one
        try:
            if '.' in answer:
                answer = float(answer)
            else:
                answer = int(answer)
        except ValueError:
            pass
        
        logger.info(f"   ✓ API result: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ API call failed: {e}")
        return None


async def llm_driven_text_extraction(
    llm_client,
    question: str,
    analysis: dict,
    context: dict
) -> Optional[str]:
    """
    Use LLM to extract specific text/values from context.
    """
    
    # Build context string
    context_parts = []
    if context.get('audio_transcript'):
        context_parts.append(f"AUDIO TRANSCRIPT:\n{context['audio_transcript']}")
    if context.get('pdf_text'):
        context_parts.append(f"PDF TEXT:\n{context['pdf_text'][:4000]}")
    if context.get('webpage_text'):
        context_parts.append(f"WEBPAGE TEXT:\n{context['webpage_text'][:3000]}")
    if context.get('scraped_page'):
        context_parts.append(f"SCRAPED CONTENT:\n{context['scraped_page'][:3000]}")
    if context.get('dominant_color'):
        context_parts.append(f"IMAGE DOMINANT COLOR: {context['dominant_color']}")
    
    context_str = "\n\n".join(context_parts) if context_parts else "No context available"
    
    extract_prompt = f"""Extract the answer from the given context.

QUESTION: {question}

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}
KEY VALUES TO EXTRACT: {analysis.get('key_values_to_extract', [])}
EXPECTED FORMAT: {analysis.get('answer_format', 'string')}

CONTEXT:
{context_str}

Return ONLY the extracted answer value (no explanations).
If the answer is a path, return just the path.
If the answer is a number, return just the number.
If the answer is a color, return the hex code."""

    try:
        answer = await llm_client.generate(extract_prompt, max_tokens=300, temperature=0.1)
        answer = answer.strip()
        
        # Clean common prefixes
        for prefix in ["Answer:", "The answer is:", "Result:"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        logger.info(f"   ✓ Extracted: {answer[:100]}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Text extraction failed: {e}")
        return None


async def llm_driven_image_analysis(
    llm_client,
    question: str,
    analysis: dict,
    context: dict
) -> Optional[str]:
    """
    Use LLM to answer questions about images.
    Uses pre-extracted image data (dominant color, description).
    """
    
    image_info = []
    if context.get('dominant_color'):
        image_info.append(f"Dominant color (hex): {context['dominant_color']}")
    if context.get('image_description'):
        image_info.append(f"Image description: {context['image_description']}")
    
    if not image_info:
        return None
    
    image_prompt = f"""Answer the image-related question using the analyzed image data.

QUESTION: {question}

IMAGE ANALYSIS:
{chr(10).join(image_info)}

EXPECTED FORMAT: {analysis.get('answer_format', 'string')}

Return ONLY the answer (e.g., just the hex color code like #rrggbb)."""

    try:
        answer = await llm_client.generate(image_prompt, max_tokens=100, temperature=0.1)
        answer = answer.strip()
        
        # If asking for color, ensure it's a valid hex
        if analysis.get('answer_format') == 'hex_color':
            if not answer.startswith('#'):
                answer = f"#{answer}"
            answer = answer[:7]  # Ensure #rrggbb format
        
        logger.info(f"   ✓ Image answer: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Image analysis failed: {e}")
        return context.get('dominant_color')  # Fallback to raw dominant color


async def llm_driven_json_transformation(
    llm_client,
    df: pd.DataFrame,
    question: str,
    analysis: dict
) -> Optional[str]:
    """
    Use LLM to transform data to JSON format.
    """
    from .data_sourcing import get_dataframe_info
    
    df_info = get_dataframe_info(df)
    
    transform_prompt = f"""Transform this data to JSON as specified.

QUESTION: {question}

DATAFRAME:
- Columns: {df_info.get('columns')}
- Sample:
{df_info.get('sample', '')}

TRANSFORMATIONS NEEDED: {analysis.get('transformations', [])}

Generate Python code that:
1. Transforms the DataFrame as needed
2. Converts to JSON array format
3. Stores result in 'answer' variable

Return ONLY Python code:"""

    try:
        code = await llm_client.generate(transform_prompt, max_tokens=500, temperature=0.1)
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```python"] else lines[1:])
        
        from .analysis import execute_analysis_code
        answer, output, error = await execute_analysis_code(code, df)
        
        if error:
            logger.warning(f"   ⚠️ JSON transformation error: {error}")
            # Fallback: simple conversion
            return df.to_json(orient='records')
        
        logger.info(f"   ✓ JSON transformation: {len(str(answer))} chars")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ JSON transformation failed: {e}")
        return df.to_json(orient='records')


async def format_answer_dynamically(
    llm_client,
    answer: Any,
    question: str,
    analysis: dict
) -> str:
    """
    Use LLM to format the answer correctly for submission.
    """
    
    expected_format = analysis.get('answer_format', 'string')
    
    # Simple cases - don't need LLM
    if answer is None:
        return analysis.get('fallback_answer', 'start')
    
    # String type - just return the string value directly
    if expected_format == 'string':
        answer_str = str(answer).strip()
        # Remove any markdown or code blocks
        if answer_str.startswith('```'):
            lines = answer_str.split('\n')
            answer_str = '\n'.join(lines[1:-1] if lines[-1].strip().startswith('```') else lines[1:])
        # Remove surrounding quotes
        if (answer_str.startswith('"') and answer_str.endswith('"')) or \
           (answer_str.startswith("'") and answer_str.endswith("'")):
            answer_str = answer_str[1:-1]
        return answer_str.strip()
    
    if expected_format == 'number':
        try:
            if isinstance(answer, float):
                # Check if question mentions decimal places
                if '2 decimal' in question.lower():
                    return str(round(answer, 2))
                return str(answer)
            return str(int(answer))
        except (ValueError, TypeError):
            pass
    
    if expected_format == 'hex_color':
        answer_str = str(answer)
        if not answer_str.startswith('#'):
            answer_str = f"#{answer_str}"
        return answer_str[:7]
    
    if expected_format == 'json':
        if isinstance(answer, str):
            return answer
        return json.dumps(answer)
    
    if expected_format == 'boolean':
        return str(answer).lower()
    
    # For complex cases, ask LLM
    format_prompt = f"""Format this answer for submission.

ORIGINAL QUESTION: {question}
COMPUTED ANSWER: {answer}
EXPECTED FORMAT: {expected_format}

Return ONLY the formatted answer ready for submission (no explanations)."""

    try:
        formatted = await llm_client.generate(format_prompt, max_tokens=200, temperature=0.1)
        formatted = formatted.strip()
        
        # Remove quotes if the answer was quoted
        if formatted.startswith('"') and formatted.endswith('"'):
            formatted = formatted[1:-1]
        if formatted.startswith("'") and formatted.endswith("'"):
            formatted = formatted[1:-1]
        
        return formatted
        
    except Exception as e:
        logger.warning(f"   ⚠️ Format failed, using raw: {e}")
        return str(answer)


async def llm_driven_network_analysis(
    llm_client,
    question: str,
    analysis: dict,
    context: dict,
    session: Any
) -> Optional[Any]:
    """
    Use LLM to solve network/graph analysis problems.
    Handles NetworkX operations, shortest paths, centrality, etc.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("   ⚠️ NetworkX not available")
        return None
    
    # Build graph from context if available
    edges_data = context.get('edges_data') or context.get('graph_data')
    
    if not edges_data and context.get('dataframe') is not None:
        # Try to convert DataFrame to edges
        df = context['dataframe']
        if len(df.columns) >= 2:
            edges_data = list(df.iloc[:, :2].values.tolist())
    
    if not edges_data:
        return None
    
    code_prompt = f"""Generate Python code to solve this network/graph problem using NetworkX.

QUESTION: {question}

AVAILABLE EDGES: {edges_data[:20]}... (showing first 20 edges)

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}

Generate ONLY Python code:
- Build the graph using NetworkX
- Perform the required analysis (shortest path, centrality, etc.)
- Store final answer in 'answer' variable

Import statements should include networkx as nx.
edges = {edges_data}

Code:"""

    try:
        code = await llm_client.generate(code_prompt, max_tokens=600, temperature=0.1)
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```python"] else lines[1:])
        
        # Execute the code
        namespace = {
            'nx': nx,
            'edges': edges_data,
            'answer': None,
            'math': math
        }
        
        exec(code, namespace)
        answer = namespace.get('answer')
        
        logger.info(f"   ✓ Network analysis result: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Network analysis failed: {e}")
        return None


async def llm_driven_geospatial(
    llm_client,
    question: str,
    analysis: dict,
    context: dict
) -> Optional[Any]:
    """
    Handle geospatial calculations like Haversine distance.
    """
    
    code_prompt = f"""Generate Python code to solve this geospatial problem.

QUESTION: {question}

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}

Generate ONLY Python code:
- Implement Haversine formula if distance calculation is needed
- Handle latitude/longitude coordinates
- Store final answer in 'answer' variable

Haversine formula reference:
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

Code:"""

    try:
        code = await llm_client.generate(code_prompt, max_tokens=500, temperature=0.1)
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```python"] else lines[1:])
        
        namespace = {
            'math': math,
            'answer': None
        }
        
        exec(code, namespace)
        answer = namespace.get('answer')
        
        logger.info(f"   ✓ Geospatial result: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ Geospatial calculation failed: {e}")
        return None


async def llm_driven_sql_analysis(
    llm_client,
    df: pd.DataFrame,
    question: str,
    analysis: dict
) -> Optional[Any]:
    """
    Use DuckDB for SQL queries on DataFrames.
    Useful when questions require SQL-style analysis.
    """
    try:
        import duckdb
    except ImportError:
        logger.warning("   ⚠️ DuckDB not available")
        return None
    
    from .data_sourcing import get_dataframe_info
    df_info = get_dataframe_info(df)
    
    sql_prompt = f"""Generate a SQL query to answer this question.

QUESTION: {question}

TABLE 'data':
- Columns: {df_info.get('columns')}
- Sample:
{df_info.get('sample', '')}

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}

Return ONLY the SQL query (SELECT statement), nothing else.
The table is named 'data'."""

    try:
        sql = await llm_client.generate(sql_prompt, max_tokens=300, temperature=0.1)
        sql = sql.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```sql"] else lines[1:])
        sql = sql.strip('`').strip()
        
        logger.info(f"   🔍 SQL: {sql}")
        
        # Execute using DuckDB
        result = duckdb.query(f"SELECT * FROM df WHERE 1=0")  # Validate df access
        result = duckdb.query(sql.replace('data', 'df'))
        answer = result.fetchall()
        
        # Return single value if single result
        if len(answer) == 1 and len(answer[0]) == 1:
            answer = answer[0][0]
        
        logger.info(f"   ✓ SQL result: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ SQL analysis failed: {e}")
        return None


async def llm_driven_css_scraping(
    llm_client,
    question: str,
    analysis: dict,
    context: dict
) -> Optional[Any]:
    """
    Use LLM to generate CSS selectors for web scraping.
    """
    from bs4 import BeautifulSoup
    
    html_content = context.get('html_content') or context.get('webpage_html', '')
    
    if not html_content:
        return None
    
    selector_prompt = f"""Generate a CSS selector to extract data from this HTML.

QUESTION: {question}

HTML SNIPPET (first 2000 chars):
{html_content[:2000]}

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}

Return ONLY valid JSON:
{{
    "selector": "<CSS selector>",
    "attribute": "<attribute to extract, or 'text' for text content>",
    "multiple": true/false
}}"""

    try:
        response = await llm_client.generate(selector_prompt, max_tokens=200, temperature=0.1)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```json"] else lines[1:])
        
        plan = json.loads(response)
        
        soup = BeautifulSoup(html_content, 'lxml')
        elements = soup.select(plan['selector'])
        
        if plan.get('attribute') == 'text':
            values = [el.get_text(strip=True) for el in elements]
        else:
            values = [el.get(plan['attribute']) for el in elements]
        
        if not plan.get('multiple'):
            answer = values[0] if values else None
        else:
            answer = values
        
        logger.info(f"   ✓ CSS scraping result: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ CSS scraping failed: {e}")
        return None


async def llm_driven_llm_task(
    llm_client,
    question: str,
    analysis: dict,
    context: dict
) -> Optional[str]:
    """
    Handle LLM-specific tasks like:
    - Prompt injection detection/crafting
    - Token counting
    - Embedding similarity
    - Function calling format
    """
    
    task_prompt = f"""Solve this LLM-related task.

QUESTION: {question}

SOLUTION STRATEGY: {analysis.get('solution_strategy', '')}

CONTEXT:
{json.dumps({k: str(v)[:500] for k, v in context.items() if v is not None}, indent=2)}

If this involves:
- Prompt injection: Craft the shortest prompt to achieve the goal
- Token counting: Estimate token count
- Function calling: Generate the JSON schema
- Embedding similarity: Explain the approach

Return ONLY the answer (no explanations)."""

    try:
        answer = await llm_client.generate(task_prompt, max_tokens=400, temperature=0.2)
        answer = answer.strip()
        
        # Clean code blocks
        if answer.startswith("```"):
            lines = answer.split("\n")
            answer = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
        
        logger.info(f"   ✓ LLM task result: {answer[:100]}")
        return answer
        
    except Exception as e:
        logger.error(f"   ❌ LLM task failed: {e}")
        return None


async def llm_driven_visualization(
    llm_client,
    df: pd.DataFrame,
    question: str,
    analysis: dict,
    session: Any
) -> Optional[str]:
    """
    Use LLM to generate visualization code and return base64 PNG.
    
    GA8 questions often ask for:
    - Bar charts (blue/green bars)
    - Scatter plots with regression lines (red dotted)
    - Histograms
    - Line charts (cumulative)
    
    Output must be base64 PNG under 100KB.
    """
    import io
    import base64
    
    df_info = {
        'columns': list(df.columns),
        'shape': df.shape,
        'sample': df.head(3).to_string()
    }
    
    chart_prompt = f"""Generate Python matplotlib code to create a chart.

QUESTION: {question}

DATAFRAME INFO:
- Shape: {df_info['shape']}
- Columns: {df_info['columns']}
- Sample:
{df_info['sample']}

REQUIREMENTS:
- Use matplotlib.pyplot as plt
- The DataFrame is already loaded as 'df'
- Save figure to a BytesIO buffer
- Set figure size to (8, 6) for good quality
- Set DPI to 100 to keep file size under 100KB
- Use specific colors if mentioned (blue, green, red, etc.)
- Add labels and title
- Store the BytesIO buffer in variable 'buffer'

Generate ONLY Python code (no explanations):

import matplotlib.pyplot as plt
import io

# Your chart code here
buffer = io.BytesIO()
# ... save to buffer

Code:"""

    try:
        code = await llm_client.generate(chart_prompt, max_tokens=600, temperature=0.1)
        
        # Clean code
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```python"] else lines[1:])
        
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        
        logger.info(f"   🎨 Generated chart code:")
        for line in code.split('\n')[:5]:
            logger.info(f"      {line}")
        
        # Execute the code
        namespace = {
            'df': df.copy(),
            'pd': pd,
            'np': __import__('numpy'),
            'plt': __import__('matplotlib.pyplot'),
            'io': io,
            'buffer': None
        }
        
        # Import matplotlib properly
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        namespace['plt'] = plt
        
        exec(code, namespace)
        
        buffer = namespace.get('buffer')
        if buffer is None:
            logger.warning("   ⚠️ No buffer produced")
            return None
        
        # Encode to base64
        buffer.seek(0)
        image_data = buffer.getvalue()
        
        # Check size (must be under 100KB)
        if len(image_data) > 100_000:
            logger.warning(f"   ⚠️ Image too large: {len(image_data)} bytes, regenerating...")
            # Try with lower DPI
            plt.figure(figsize=(6, 4), dpi=80)
            exec(code, namespace)
            buffer = namespace.get('buffer')
            if buffer:
                buffer.seek(0)
                image_data = buffer.getvalue()
        
        base64_str = base64.b64encode(image_data).decode('utf-8')
        
        # Return as data URI
        data_uri = f"data:image/png;base64,{base64_str}"
        
        logger.info(f"   ✓ Chart generated: {len(base64_str)} chars")
        return data_uri
        
    except Exception as e:
        logger.error(f"   ❌ Visualization failed: {e}")
        return None


async def solve_with_llm(
    llm_client,
    question: str,
    context: dict,
    df: Optional[pd.DataFrame],
    session: Any
) -> Any:
    """
    NEW ARCHITECTURE: Generic task classification + handler routing.
    
    This replaces 500+ lines of hardcoded if/elif chains with:
    1. LLM classifies task type dynamically
    2. Routes to appropriate generic handler
    3. Handler solves ANY question of that type
    
    No more keyword matching. No more brittle patterns.
    Works for ALL question variations, not just memorized ones.
    """
    
    logger.info("   🧠 Step 1: Dynamic Task Classification...")
    
    # Add DataFrame to context for classification
    if df is not None:
        context['dataframe'] = df
    
    # Use new classification method
    classification = await llm_client.classify_task_dynamically(question, context)
    
    task_type = classification.get('task_type', 'other')
    logger.info(f"   ✓ Task: {task_type} (confidence: {classification.get('confidence', 'N/A')})")
    logger.info(f"   ✓ Answer format: {classification.get('answer_format')}")
    logger.info(f"   ✓ Has personalization: {classification.get('has_personalization')}")
    
    # Step 2: Route to generic handler
    from .task_handlers import TASK_HANDLERS
    
    handler = TASK_HANDLERS.get(task_type, TASK_HANDLERS['other'])
    logger.info(f"   🎯 Routing to: {handler.__name__}")
    
    try:
        answer = await handler(
            question=question,
            context=context,
            classification=classification,
            session=session
        )
    except Exception as e:
        logger.error(f"   ❌ Handler {handler.__name__} failed: {e}")
        # Fallback to generic LLM answer
        answer = await TASK_HANDLERS['other'](
            question=question,
            context=context,
            classification=classification,
            session=session
        )
    
    # FIX #3: Handler already returns formatted answer - no post-processing needed
    # Removed: analyze_strategy(), extract_answer_from_analysis(), process_answer()
    # Time saved: 54 seconds
    
    if answer is not None:
        logger.info(f"   ✓ Final answer: {str(answer)[:100]}")
        return answer
    
    # Ultimate fallback
    logger.warning("   ⚠️  No answer generated, using fallback")
    return "start"
