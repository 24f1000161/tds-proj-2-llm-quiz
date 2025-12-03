"""
LLM-driven analysis module.

This module contains all LLM-driven logic for solving quiz questions.
NO hardcoded patterns - all decisions go through LLM.

Covers ALL TDS course topics:
- Data Analysis: pandas, DuckDB, SQL, aggregation, filtering, grouping, joins
- Web Scraping: CSS selectors, BeautifulSoup, JavaScript rendering
- API Calls: GitHub API, REST APIs, authentication, rate limits
- Command Generation: uv, git, curl, wget, bash, shell commands
- File Processing: PDF, CSV, JSON, JSONL, Parquet, Excel
- Network Analysis: NetworkX, shortest paths, graph operations, centrality
- Geospatial: Haversine distance, coordinates, lat/long, mapping
- Image Analysis: dominant color, compression, base64 encoding
- Audio/Video: transcription, YouTube, audio extraction
- LLM Tasks: prompt injection, embeddings, function calling, tokenization
- Statistical: correlation, regression, forecasting, outlier detection
"""

import json
import re
import math
from typing import Any, Optional, List, Dict
import pandas as pd

from .logging_utils import logger


# Course topics knowledge base for better question understanding
TDS_TOPICS = {
    "data_sourcing": ["API", "scraping", "wget", "curl", "httpx", "requests", "download", "fetch"],
    "data_preparation": ["clean", "transform", "filter", "merge", "join", "pandas", "DuckDB", "OpenRefine", "regex"],
    "data_analysis": ["sum", "count", "average", "mean", "median", "correlation", "regression", "aggregate", "groupby"],
    "data_visualization": ["chart", "plot", "graph", "seaborn", "matplotlib", "base64", "image"],
    "git_github": ["git", "commit", "push", "pull", "branch", "clone", "repository", "GitHub"],
    "llm": ["LLM", "GPT", "Gemini", "prompt", "token", "embedding", "function calling"],
    "network": ["graph", "node", "edge", "shortest path", "NetworkX", "centrality", "degree"],
    "geospatial": ["latitude", "longitude", "coordinate", "distance", "haversine", "location", "map"],
    "deployment": ["Docker", "Vercel", "Hugging Face", "FastAPI", "deploy", "server"],
    "shell": ["bash", "terminal", "command", "uv", "pip", "npm", "shell script"],
}


async def analyze_question_deeply(llm_client, question_text: str, context: dict) -> dict:
    """
    Use LLM to deeply analyze the question and determine the solution strategy.
    
    Returns a structured analysis with task type, answer format, and solution strategy.
    
    Task types:
    - data_analysis: pandas/SQL analysis on DataFrames
    - web_scrape: CSS selectors, BeautifulSoup extraction
    - command_generation: git, uv, curl, wget, bash commands
    - api_call: GitHub API, REST API calls
    - text_extraction: Extract values from PDF, text, webpage
    - audio_transcription: Audio content extraction
    - image_analysis: Color analysis, image questions
    - network_analysis: NetworkX, graph analysis, shortest paths
    - geospatial: Haversine distance, coordinates
    - llm_task: Prompt injection, embeddings, function calling
    - json_transform: Convert data to JSON format
    - intro_page: Quiz intro/start page
    """
    
    # Build context summary for LLM
    context_summary = []
    if context.get('audio_transcript'):
        context_summary.append(f"Audio transcript available: {context['audio_transcript'][:200]}...")
    if context.get('dominant_color'):
        context_summary.append(f"Image analyzed - dominant color: {context['dominant_color']}")
    if context.get('pdf_text'):
        context_summary.append(f"PDF text available: {len(context['pdf_text'])} chars")
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
    
    context_str = "\n".join(context_summary) if context_summary else "No additional context available"
    
    analysis_prompt = f"""Analyze this quiz question and determine the best solution approach.

QUESTION:
{question_text}

AVAILABLE CONTEXT:
{context_str}

Return ONLY valid JSON with this structure:
{{
    "task_type": "data_analysis|web_scrape|command_generation|api_call|text_extraction|audio_transcription|image_analysis|network_analysis|geospatial|llm_task|json_transform|intro_page",
    "answer_format": "number|string|json|boolean|command|path|hex_color|list",
    "solution_strategy": "<describe exactly how to solve this step by step>",
    "data_needed": ["<list what data is needed>"],
    "key_values_to_extract": ["<specific values to find>"],
    "transformations": ["<any transformations needed>"],
    "personalization": {{
        "uses_email": true/false,
        "offset_calculation": "<formula if any, e.g., 'len(email) % 5'>"
    }},
    "confidence": 0.0-1.0,
    "fallback_answer": "<if we can't compute, what to submit>"
}}

Task type guidance:
- data_analysis: Questions about filtering, aggregating, counting, summing data in CSV/DataFrame
- web_scrape: Questions needing CSS selectors or HTML parsing
- command_generation: Questions asking for git, uv, curl, wget, bash commands
- api_call: Questions about GitHub repos, REST APIs, counting files
- text_extraction: Questions asking to find specific values in text/PDF
- audio_transcription: Questions about audio content
- image_analysis: Questions about image colors, visual content
- network_analysis: Questions about graphs, shortest paths, connections
- geospatial: Questions about distances, coordinates, locations
- llm_task: Questions about prompts, tokens, embeddings, function calling
- json_transform: Questions asking for JSON output format
- intro_page: Quiz intro that just needs "start" answer

Be specific about the solution strategy."""

    try:
        response = await llm_client.generate(analysis_prompt, max_tokens=800, temperature=0.1)
        
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```json"] else lines[1:])
        
        analysis = json.loads(response)
        logger.info(f"   📊 Question analysis: task={analysis.get('task_type')}, format={analysis.get('answer_format')}")
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

DATAFRAME INFO:
- Shape: {df_info.get('shape')}
- Columns: {df_info.get('columns')}
- Sample data:
{df_info.get('sample', '')}

SOLUTION GUIDANCE: {guidance}
TRANSFORMATIONS NEEDED: {transformations}

{"PERSONALIZATION: Email length is " + str(len(session.email)) + ", use for offset calculation: " + personalization.get('offset_calculation', '') if personalization.get('uses_email') else ""}

Generate ONLY Python code (no explanations):
- Use variable 'df' for the dataframe
- Store final answer in variable 'answer'
- Handle edge cases (NaN, type conversion)
- Apply any required formatting (decimals, JSON, etc.)

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
{json.dumps({k: str(v)[:500] for k, v in context.items() if v}, indent=2)}

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


async def solve_with_llm(
    llm_client,
    question: str,
    context: dict,
    df: Optional[pd.DataFrame],
    session: Any
) -> Any:
    """
    Main entry point for LLM-driven question solving.
    
    1. Analyze the question deeply
    2. Route to appropriate handler based on task type
    3. Format and return answer
    
    Handles ALL TDS course topics:
    - Data Analysis (pandas, SQL, DuckDB)
    - Web Scraping (CSS selectors, BeautifulSoup)
    - API Calls (GitHub, REST APIs)
    - Commands (git, uv, curl, wget, bash)
    - Network Analysis (NetworkX, graphs)
    - Geospatial (Haversine, coordinates)
    - Image Analysis (dominant color)
    - Audio Transcription
    - LLM Tasks (prompt injection, function calling)
    """
    
    logger.info("   🧠 Analyzing question with LLM...")
    
    # Step 1: Deep question analysis
    analysis = await analyze_question_deeply(llm_client, question, context)
    
    task_type = analysis.get('task_type', 'llm_task')
    logger.info(f"   📋 Task type: {task_type}")
    logger.info(f"   📋 Strategy: {analysis.get('solution_strategy', 'N/A')[:100]}")
    
    # Step 2: Route to appropriate handler
    answer = None
    
    # Check for intro/start page
    if task_type == 'intro_page':
        answer = analysis.get('fallback_answer', 'start')
        logger.info(f"   ✓ Intro page detected, using: {answer}")
    
    # Data Analysis (pandas/SQL)
    elif task_type == 'data_analysis' and df is not None:
        # Try SQL first for complex queries
        if any(kw in question.lower() for kw in ['sql', 'query', 'select', 'where', 'group by', 'having']):
            answer = await llm_driven_sql_analysis(llm_client, df, question, analysis)
        
        # Fall back to pandas
        if answer is None:
            answer = await llm_driven_data_analysis(
                llm_client, df, context, question, analysis, session
            )
    
    # Command generation (git, uv, curl, bash)
    elif task_type == 'command_generation':
        answer = await llm_driven_command_generation(
            llm_client, question, analysis, session
        )
    
    # API calls
    elif task_type == 'api_call':
        answer = await llm_driven_api_call(
            llm_client, question, analysis, context, session
        )
    
    # Image analysis
    elif task_type == 'image_analysis':
        answer = await llm_driven_image_analysis(
            llm_client, question, analysis, context
        )
    
    # Audio transcription
    elif task_type == 'audio_transcription':
        answer = context.get('audio_transcript', '').strip()
        if answer:
            logger.info(f"   ✓ Using audio transcript: {answer[:50]}")
    
    # Network/Graph analysis
    elif task_type == 'network_analysis':
        answer = await llm_driven_network_analysis(
            llm_client, question, analysis, context, session
        )
    
    # Geospatial calculations
    elif task_type == 'geospatial':
        answer = await llm_driven_geospatial(
            llm_client, question, analysis, context
        )
    
    # Web scraping with CSS selectors
    elif task_type == 'web_scrape':
        answer = await llm_driven_css_scraping(
            llm_client, question, analysis, context
        )
        if answer is None:
            answer = await llm_driven_text_extraction(
                llm_client, question, analysis, context
            )
    
    # LLM-specific tasks
    elif task_type == 'llm_task':
        answer = await llm_driven_llm_task(
            llm_client, question, analysis, context
        )
    
    # Text extraction (PDF, webpage, etc.)
    elif task_type == 'text_extraction':
        answer = await llm_driven_text_extraction(
            llm_client, question, analysis, context
        )
    
    # JSON transformation
    elif task_type == 'json_transform' and df is not None:
        answer = await llm_driven_json_transformation(
            llm_client, df, question, analysis
        )
    
    # Fallback: Use LLM directly with all context
    if answer is None:
        logger.info("   🤖 Using LLM fallback...")
        answer = await llm_driven_text_extraction(
            llm_client, question, analysis, context
        )
    
    # Step 3: Format answer
    if answer is not None:
        formatted = await format_answer_dynamically(
            llm_client, answer, question, analysis
        )
        return formatted
    
    # Ultimate fallback
    return analysis.get('fallback_answer', 'start')
