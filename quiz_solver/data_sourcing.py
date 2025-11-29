"""
Data sourcing, fetching, and parsing module.
"""

import io
import random
import base64
from typing import Any, Optional
import aiohttp
import pandas as pd
import pypdf
import pdfplumber

from .config import settings
from .logging_utils import logger, log_step


# User-Agent rotation pool for anti-blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]


def get_random_user_agent() -> str:
    """Get a random User-Agent for request rotation."""
    return random.choice(USER_AGENTS)


def get_request_headers(url: str = "") -> dict[str, str]:
    """Get headers with random User-Agent for anti-blocking."""
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    
    # Add Referer for specific sites that check it
    if "wikipedia.org" in url:
        headers["Referer"] = "https://www.google.com/"
    
    return headers


async def download_file(url: str, max_retries: int = 3) -> bytes:
    """Download file with retry logic and User-Agent rotation."""
    
    for attempt in range(max_retries):
        try:
            headers = get_request_headers(url)
            timeout = aiohttp.ClientTimeout(total=settings.timeouts.download_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=False) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    else:
                        logger.warning(f"Download got status {resp.status} for {url}")
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception(f"Failed to download {url} after {max_retries} attempts")


async def download_and_parse_pdf(url: str) -> dict[str, Any]:
    """Download and extract content from PDF."""
    
    file_bytes = await download_file(url)
    
    extracted: dict[str, Any] = {
        "pages": {},
        "all_tables": [],
        "all_text": ""
    }
    
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                
                extracted["pages"][page_num] = {
                    "text": text,
                    "tables": tables
                }
                extracted["all_tables"].extend(tables)
                extracted["all_text"] += text + "\n"
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}, trying pypdf")
        # Fallback to pypdf
        with io.BytesIO(file_bytes) as f:
            pdf = pypdf.PdfReader(f)
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                extracted["pages"][page_num] = {"text": text, "tables": []}
                extracted["all_text"] += text + "\n"
    
    logger.info(f"Extracted {len(extracted['pages'])} pages from PDF")
    return extracted


async def download_and_parse_file(url: str) -> Any:
    """Download and parse CSV, JSON, or Excel files."""
    
    file_bytes = await download_file(url)
    
    if url.endswith('.csv'):
        # Try reading with header first
        df = pd.read_csv(io.BytesIO(file_bytes))
        
        # Check if the header looks like a data value (e.g., a number)
        # If so, read without header
        first_col = str(df.columns[0])
        if first_col.isdigit() or (first_col.replace('.', '', 1).isdigit()):
            logger.info(f"CSV header looks like data, re-reading without header")
            df = pd.read_csv(io.BytesIO(file_bytes), header=None)
            df.columns = [f'value_{i}' for i in range(len(df.columns))]
        
        logger.info(f"Parsed CSV: {df.shape}, columns: {list(df.columns)}")
        return df
    
    elif url.endswith('.json'):
        import json
        data = json.loads(file_bytes.decode('utf-8'))
        if isinstance(data, list):
            df = pd.DataFrame(data)
            logger.info(f"Parsed JSON to DataFrame: {df.shape}")
            return df
        return data
    
    elif url.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(file_bytes))
        logger.info(f"Parsed Excel: {df.shape}")
        return df
    
    else:
        # Return raw bytes
        return file_bytes


async def call_api(url: str) -> Any:
    """Call an API endpoint and return JSON response."""
    
    timeout = aiohttp.ClientTimeout(total=settings.timeouts.download_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, ssl=False) as resp:
            if resp.status == 200:
                content_type = resp.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return await resp.json()
                else:
                    return await resp.text()
            else:
                raise Exception(f"API call failed with status {resp.status}")


async def scrape_webpage(url: str) -> str:
    """Scrape webpage content."""
    
    headers = get_request_headers(url)
    timeout = aiohttp.ClientTimeout(total=settings.timeouts.download_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers, ssl=False) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                raise Exception(f"Scrape failed with status {resp.status}")


async def scrape_webpage_static(url: str) -> dict[str, Any]:
    """
    Static scraping using BeautifulSoup for non-JS sites.
    Returns structured data: text, tables, lists, key-values.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("BeautifulSoup not available, falling back to raw scraping")
        return {"text": await scrape_webpage(url), "tables": [], "lists": []}
    
    headers = get_request_headers(url)
    timeout = aiohttp.ClientTimeout(total=settings.timeouts.download_timeout)
    
    result: dict[str, Any] = {
        "html": "",
        "text": "",
        "tables": [],
        "lists": [],
        "key_values": {},
    }
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    logger.warning(f"Static scrape got status {resp.status}")
                    return result
                
                html = await resp.text()
        
        # Parse with BeautifulSoup - try lxml first, fallback to html.parser
        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, nav, header, footer elements
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()
        
        result["html"] = str(soup)
        result["text"] = soup.get_text(separator='\n', strip=True)
        
        # Extract tables to DataFrames
        result["tables"] = extract_html_tables(soup)
        
        # Extract lists
        result["lists"] = extract_list_items(soup)
        
        # Extract key-value pairs
        result["key_values"] = extract_key_value_pairs(soup)
        
        logger.info(f"Static scraped {url}: {len(result['text'])} chars, {len(result['tables'])} tables")
        return result
    
    except Exception as e:
        logger.error(f"Static scraping failed: {e}")
        return result


def extract_html_tables(soup) -> list[pd.DataFrame]:
    """Extract HTML tables to pandas DataFrames."""
    tables = []
    
    for table in soup.find_all('table'):
        try:
            html_str = str(table)
            dfs = pd.read_html(io.StringIO(html_str))
            if dfs:
                tables.extend(dfs)
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")
            # Manual parsing fallback
            try:
                rows = table.find_all('tr')
                if not rows:
                    continue
                
                data = []
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    data.append([cell.get_text(strip=True) for cell in cells])
                
                if data:
                    df = pd.DataFrame(data[1:], columns=data[0] if data else None)
                    tables.append(df)
            except Exception:
                pass
    
    return tables


def extract_list_items(soup) -> list[dict[str, Any]]:
    """Extract <ul>/<ol> lists to Python lists."""
    result = []
    
    for list_tag in soup.find_all(['ul', 'ol']):
        list_type = "ordered" if list_tag.name == 'ol' else "unordered"
        items = []
        
        for li in list_tag.find_all('li', recursive=False):
            text = li.get_text(strip=True)
            if text:
                items.append(text)
        
        if items:
            result.append({
                "type": list_type,
                "items": items
            })
    
    return result


def extract_key_value_pairs(soup) -> dict[str, str]:
    """Extract key-value pairs from definition lists and patterns."""
    import re
    
    pairs: dict[str, str] = {}
    
    # Pattern 1: Definition lists
    for dl in soup.find_all('dl'):
        dts = dl.find_all('dt')
        dds = dl.find_all('dd')
        for dt, dd in zip(dts, dds):
            key = dt.get_text(strip=True)
            value = dd.get_text(strip=True)
            if key and value:
                pairs[key] = value
    
    # Pattern 2: Two-column tables (key in col 1, value in col 2)
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                if key and value and len(key) < 50:
                    pairs[key] = value
    
    # Pattern 3: Label: Value in text (e.g., "Price: $99.99")
    text = soup.get_text()
    kv_pattern = r'([A-Za-z][A-Za-z\s]{1,30}):\s*([^\n:]{1,100})'
    matches = re.findall(kv_pattern, text)
    for key, value in matches:
        key = key.strip()
        value = value.strip()
        if key and value and len(key) < 50 and key not in pairs:
            pairs[key] = value
    
    return pairs


async def transcribe_audio(url: str) -> str:
    """
    Transcribe audio file using Gemini via aipipe.
    Supports: .mp3, .wav, .ogg, .m4a, .flac, .opus
    """
    import httpx
    
    # Download audio file
    audio_bytes = await download_file(url)
    logger.info(f"Downloaded audio file: {len(audio_bytes)} bytes")
    
    # Determine mime type from extension
    ext = url.split('.')[-1].lower()
    mime_types = {
        'mp3': 'audio/mp3',
        'wav': 'audio/wav',
        'ogg': 'audio/ogg',
        'm4a': 'audio/mp4',
        'flac': 'audio/flac',
        'opus': 'audio/opus',
    }
    mime_type = mime_types.get(ext, 'audio/mp3')
    
    # Encode as base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Use Gemini via aipipe's direct Gemini endpoint (supports audio)
    if settings.llm.aipipe_token:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://aipipe.org/geminiv1beta/models/gemini-2.0-flash:generateContent",
                    headers={
                        "x-goog-api-key": settings.llm.aipipe_token,
                        "Content-Type": "application/json"
                    },
                    json={
                        "contents": [{
                            "parts": [
                                {"text": "Transcribe this audio file completely. Return only the transcription text."},
                                {"inline_data": {"mime_type": mime_type, "data": audio_b64}}
                            ]
                        }],
                        "generationConfig": {"maxOutputTokens": 4000}
                    }
                )
                response.raise_for_status()
                data = response.json()
                transcript = data["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"Audio transcribed via aipipe Gemini: {len(transcript)} chars")
                return transcript
        except Exception as e:
            logger.warning(f"Aipipe Gemini audio transcription failed: {e}")
    
    # Fallback: try direct Gemini API with separate key
    if settings.llm.gemini_api_key:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={
                        "x-goog-api-key": settings.llm.gemini_api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "contents": [{
                            "parts": [
                                {"text": "Transcribe this audio file completely. Return only the transcription text."},
                                {"inline_data": {"mime_type": mime_type, "data": audio_b64}}
                            ]
                        }],
                        "generationConfig": {"maxOutputTokens": 4000}
                    }
                )
                response.raise_for_status()
                data = response.json()
                transcript = data["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"Audio transcribed via direct Gemini: {len(transcript)} chars")
                return transcript
        except Exception as e:
            logger.warning(f"Direct Gemini audio transcription failed: {e}")
    
    logger.error("All audio transcription methods failed")
    return ""


def is_audio_url(url: str) -> bool:
    """Check if URL is an audio file."""
    audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac', '.wma', '.opus']
    return any(url.lower().endswith(ext) for ext in audio_extensions)


async def fetch_all_data_sources(data_sources: list[str], session: Any) -> dict[str, Any]:
    """Fetch data from all identified sources."""
    
    fetched_data: dict[str, Any] = {}
    
    for source_url in data_sources:
        try:
            if source_url.endswith('.pdf'):
                fetched_data[source_url] = await download_and_parse_pdf(source_url)
            elif source_url.endswith(('.csv', '.json', '.xlsx', '.xls')):
                fetched_data[source_url] = await download_and_parse_file(source_url)
            elif is_audio_url(source_url):
                # Handle audio files with transcription
                transcript = await transcribe_audio(source_url)
                fetched_data[source_url] = {"transcript": transcript, "type": "audio"}
            elif '/api/' in source_url or source_url.endswith('/data'):
                fetched_data[source_url] = await call_api(source_url)
            else:
                # Try static scraping first (faster), then fall back to raw
                scraped = await scrape_webpage_static(source_url)
                if scraped.get("text") and len(scraped["text"]) > 50:
                    fetched_data[source_url] = scraped
                else:
                    # Fallback to raw scraping
                    fetched_data[source_url] = await scrape_webpage(source_url)
            
            log_step(session, "data_fetched", {
                "source": source_url,
                "status": "success"
            })
        
        except Exception as e:
            logger.error(f"Failed to fetch {source_url}: {e}")
            log_step(session, "data_fetch_failed", {
                "source": source_url,
                "error": str(e)
            })
    
    return fetched_data


def clean_and_prepare_data(raw_data: Any, question_context: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Convert raw data into clean pandas DataFrame."""
    
    df: Optional[pd.DataFrame] = None
    
    # Handle different input types
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    
    elif isinstance(raw_data, str):
        # Try parsing as JSON first
        try:
            import json
            data = json.loads(raw_data)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
        except Exception:
            # Try parsing as CSV
            try:
                df = pd.read_csv(io.StringIO(raw_data))
            except Exception:
                logger.warning("Could not parse string data")
                return None
    
    elif isinstance(raw_data, dict):
        # Check if it's PDF extracted data
        if 'all_tables' in raw_data and raw_data['all_tables']:
            # Take the first table
            first_table = raw_data['all_tables'][0]
            if first_table:
                df = pd.DataFrame(first_table[1:], columns=first_table[0] if first_table else None)
        elif 'pages' in raw_data:
            # Extract text data
            all_text = raw_data.get('all_text', '')
            logger.info(f"PDF text content: {all_text[:200]}...")
            return None  # Return None for text-only PDFs
        else:
            df = pd.DataFrame([raw_data])
    
    elif isinstance(raw_data, list):
        df = pd.DataFrame(raw_data)
    
    else:
        logger.warning(f"Unknown data type: {type(raw_data)}")
        return None
    
    if df is None or df.empty:
        return None
    
    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
    
    # Infer and convert column types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass  # Keep as string
    
    # Handle missing values - but don't drop them, just log
    na_count = df.isna().sum().sum()
    if na_count > 0:
        logger.info(f"Found {na_count} NA values in DataFrame")
        # For numeric columns, keep NaN (will be excluded from sum/agg)
        # For string columns, fill with empty string
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("")
    
    # DON'T remove duplicates - they may be valid data for aggregation
    # df = df.drop_duplicates()
    
    logger.info(f"Cleaned DataFrame: {df.shape}, columns: {list(df.columns)}")
    return df


def get_dataframe_info(df: pd.DataFrame) -> dict[str, Any]:
    """Get DataFrame information for LLM analysis."""
    
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        "sample": df.head(5).to_string()
    }
