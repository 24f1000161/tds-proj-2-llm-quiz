# LLM Quiz Solver Agent - Production Implementation Prompt

You are a production-ready data science agent responsible for solving automated quizzes through an intelligent, multi-stage pipeline. Your goal is to receive quiz URLs, analyze questions, extract/process data, perform analysis, and submit correct answers within a 3-minute window.

## Core Responsibility

You will be called by an API endpoint that receives POST requests with the following structure:
```json
{
  "email": "your email",
  "secret": "your secret",
  "url": "https://example.com/quiz-834"
}
```

Your job: 
1. Verify the secret matches what was provided in the Google Form (respond with 403 if invalid)
2. Visit the URL, solve the quiz, and submit the answer to the correct endpoint
3. Handle chained quizzes (follow new URLs until quiz is complete)
4. All operations must complete within 3 minutes of original POST

---

## LLM Model Strategy: Dual-Model Approach for Token Efficiency

### Primary Model: GPT-5-Nano (via aipipe)
- **Use for**: Question parsing, task classification, simple data analysis interpretation
- **Why**: Ultra-fast, lowest latency, sufficient for structured reasoning
- **Token consumption**: Minimal (~200-500 tokens per call)
- **Timeout**: 5 seconds
- **Cost**: Cheapest tier for aipipe

### Fallback Model: Gemini 2.5 Flash
- **Use for**: Complex reasoning, multi-step analysis guidance, ambiguous question interpretation
- **Why**: Cost-effective alternative when aipipe tokens running low, strong reasoning capabilities
- **Token consumption**: Moderate (~300-1000 tokens per call)
- **Timeout**: 8 seconds
- **Setup**: Google Generative AI SDK or direct API

### Token Budget Management
```python
# Pseudo-config
BUDGET_CONFIG = {
    "aipipe_monthly_tokens": 1_000_000,  # Adjust based on plan
    "gemini_fallback_tokens": 2_000_000,  # Separate budget
    "token_tracking_db": "track_llm_usage",  # Log every call
    "auto_switch_threshold": 0.75,  # Switch models at 75% aipipe consumption
    "emergency_bypass": 0.90,  # Use only Gemini after 90%
}
```

### Model Selection Logic
```
1. Check aipipe token usage
2. If < 75% threshold → Use GPT-5-Nano
3. If 75-90% threshold → Hybrid (prefer Gemini for non-critical steps)
4. If > 90% threshold → Use Gemini exclusively
5. On aipipe timeout/error → Auto-fallback to Gemini
6. Log every model selection and reason
```

---

## API Endpoint Implementation (HTTP Layer)

### Endpoint Handler Architecture
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import json

app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# Store student secrets (loaded from secure config)
VALID_SECRETS = {
    "student1_email@example.com": "secret_key_1",
    "student2_email@example.com": "secret_key_2",
}

@app.post("/api/quiz")
async def handle_quiz_request(request: QuizRequest):
    """
    Main API endpoint for quiz requests
    
    Validates request, initiates quiz solving pipeline
    """
    try:
        # 1. VALIDATE REQUEST JSON
        if not request.email or not request.secret or not request.url:
            return {"error": "Missing required fields"}, 400
        
        # 2. VERIFY SECRET
        if request.email not in VALID_SECRETS:
            logger.warning(f"Invalid email attempt: {request.email}")
            raise HTTPException(status_code=403, detail="Invalid email or secret")
        
        if VALID_SECRETS[request.email] != request.secret:
            logger.warning(f"Invalid secret for {request.email}")
            raise HTTPException(status_code=403, detail="Invalid email or secret")
        
        # 3. LOG REQUEST
        logger.info(f"Valid request from {request.email} for {request.url}")
        
        # 4. RESPOND WITH 200 OK
        response = {
            "status": "accepted",
            "quiz_id": generate_quiz_id(request.url),
            "timestamp": datetime.now().isoformat()
        }
        
        # 5. LAUNCH ASYNC QUIZ SOLVER (don't wait for completion)
        # This starts the pipeline in background
        asyncio.create_task(
            solve_quiz_pipeline(
                email=request.email,
                secret=request.secret,
                initial_url=request.url,
                deadline=time.time() + 180  # 3 minutes
            )
        )
        
        return response
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Error Response Codes
```python
# Respond with HTTP 400 for invalid JSON
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.warning(f"Invalid JSON: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid JSON payload"}
    )

# Respond with HTTP 403 for invalid secret
# (Already handled in endpoint above)

# All successful requests return HTTP 200
```

---

## Pipeline Architecture

### Stage 1: Environment & Initialization

**1A: Session State Creation**
```python
async def create_session_from_request(email, secret, initial_url, deadline):
    """Initialize session state for quiz solving"""
    
    session = {
        # Request info
        "email": email,
        "secret": secret,
        "initial_url": initial_url,
        
        # Timing
        "start_time": time.time(),
        "deadline": deadline,  # time.time() + 180 seconds
        "deadline_safety_buffer": 10,  # Reserve 10 seconds for final submission
        
        # Quiz tracking
        "quiz_id": None,
        "question_text": None,
        "question_id": None,
        "submit_url": None,
        "expected_format": None,
        
        # Data & Analysis
        "data_sources": [],
        "raw_data": None,
        "cleaned_data": None,
        "analysis_result": None,
        "final_answer": None,
        
        # Attempt tracking
        "current_url": initial_url,
        "quiz_chain": [initial_url],
        "submission_attempts": [],
        
        # Logs
        "llm_calls_log": [],
        "error_log": [],
        "audit_trail": []
    }
    
    return session
```

**1B: Browser Setup for Vue.js/Modern Frameworks**
```python
from playwright.async_api import async_playwright

async def launch_browser():
    """Initialize headless browser with optimal settings"""
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',  # Reduce memory usage
            '--disable-gpu'
        ]
    )
    
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        ignore_https_errors=True,
        viewport={"width": 1280, "height": 720}
    )
    
    page = await context.new_page()
    
    # Set timeouts
    page.set_default_timeout(30000)  # 30 seconds
    page.set_default_navigation_timeout(30000)
    
    return playwright, browser, context, page
```

**1C: LLM Client Initialization**
```python
async def initialize_llm_clients():
    """Setup both aipipe and Gemini clients"""
    
    # Aipipe client for GPT-5-Nano
    aipipe_client = AipipeClient(
        api_key=os.getenv("AIPIPE_API_KEY"),
        base_url=os.getenv("AIPIPE_BASE_URL", "https://api.aipipe.com")
    )
    
    # Gemini client for fallback
    gemini_client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Token tracking
    token_tracker = {
        "aipipe_used": 0,
        "aipipe_total": int(os.getenv("AIPIPE_TOKEN_LIMIT", "1000000")),
        "gemini_used": 0,
        "gemini_total": int(os.getenv("GEMINI_TOKEN_LIMIT", "2000000"))
    }
    
    return aipipe_client, gemini_client, token_tracker
```

---

### Stage 2: JavaScript-Rendered Page Navigation (Vue.js & Framework Support)

**2A: Navigate to Quiz URL**
```python
async def navigate_to_quiz(page, url, session):
    """Navigate to quiz URL and wait for full rendering"""
    
    try:
        # Navigate with timeout
        response = await page.goto(
            url,
            wait_until="networkidle",  # Wait for network to be idle
            timeout=30000
        )
        
        if response.status != 200:
            logger.warning(f"Got status {response.status} for {url}")
            raise Exception(f"HTTP {response.status} from {url}")
        
        # Log navigation
        log_step(session, "navigation_complete", {
            "url": url,
            "status": response.status
        })
        
    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        raise
```

**2B: SPA (Single Page Application) Rendering - Vue.js/React/Angular**
```python
async def wait_for_spa_rendering(page, session):
    """
    Wait for modern framework (Vue/React/Angular) to render
    Multiple strategies for robust detection
    """
    
    FRAMEWORK_CHECKS = {
        "vue": {
            "indicators": ["__VUE__", "__NUXT__", "data-server-rendered"],
            "wait_selector": "[data-v-app]",
            "detection": "typeof window !== 'undefined' && window.__VUE_DEVTOOLS_GLOBAL_HOOK__ !== undefined"
        },
        "react": {
            "indicators": ["__REACT_DEVTOOLS__"],
            "wait_selector": "[data-reactroot]",
            "detection": "typeof window !== 'undefined' && window.__REACT_DEVTOOLS_GLOBAL_HOOK__ !== undefined"
        },
        "angular": {
            "indicators": ["ng-version"],
            "wait_selector": "[ng-app]",
            "detection": "typeof window !== 'undefined' && window.ng !== undefined"
        }
    }
    
    # Strategy 1: Check for framework-specific DOM elements
    try:
        detected_framework = None
        for framework, config in FRAMEWORK_CHECKS.items():
            try:
                await page.wait_for_selector(config["wait_selector"], timeout=5000)
                detected_framework = framework
                logger.info(f"Detected {framework} framework")
                break
            except:
                pass
        
        # Strategy 2: Framework detection via JS
        if not detected_framework:
            for framework, config in FRAMEWORK_CHECKS.items():
                try:
                    result = await page.evaluate(config["detection"])
                    if result:
                        detected_framework = framework
                        logger.info(f"Detected {framework} via JS")
                        break
                except:
                    pass
    
    except Exception as e:
        logger.warning(f"Framework detection failed: {e}")
    
    # Strategy 3: Wait for content in #result div
    try:
        await page.wait_for_function(
            """() => {
                const element = document.querySelector("#result");
                return element && element.textContent.trim().length > 0;
            }""",
            timeout=30000
        )
        logger.info("Content detected in #result div")
    except:
        logger.warning("Timeout waiting for #result content")
    
    # Strategy 4: Custom MutationObserver for any DOM changes
    try:
        await page.evaluate("""() => {
            return new Promise((resolve) => {
                const observer = new MutationObserver(() => {
                    const resultDiv = document.querySelector("#result");
                    if (resultDiv && resultDiv.textContent.trim().length > 100) {
                        observer.disconnect();
                        resolve(true);
                    }
                });
                observer.observe(document.body, {
                    attributes: true,
                    childList: true,
                    subtree: true,
                    characterData: true
                });
                // Fallback timeout
                setTimeout(() => { observer.disconnect(); resolve(true); }, 30000);
            });
        }""")
        logger.info("SPA content loaded via MutationObserver")
    except Exception as e:
        logger.warning(f"MutationObserver timeout: {e}")
    
    # Wait for network to settle
    await page.wait_for_load_state("networkidle")
```

**2C: Extract & Decode Content**
```python
async def extract_and_decode_content(page, session):
    """Extract all content from page and handle base64 decoding"""
    
    # Get full page HTML
    html_content = await page.content()
    
    # Find and execute all inline scripts that might decode content
    scripts = await page.locator("script:not([src])").all()
    
    for script in scripts:
        try:
            script_text = await script.text_content()
            
            # Check if script contains decoding operations
            if any(keyword in script_text for keyword in ["atob", "decode", "innerHTML", "textContent"]):
                logger.debug(f"Executing decoder script: {script_text[:100]}...")
                # Script likely already executed during page load
                # But double-check by re-evaluating it
                try:
                    await page.evaluate(script_text)
                except:
                    pass  # Script may have already run
        except:
            pass
    
    # Get final rendered HTML after all JS execution
    final_html = await page.content()
    final_text = await page.evaluate("document.body.innerText")
    
    # Get the #result div specifically (where decoded content usually appears)
    try:
        result_content = await page.locator("#result").inner_html()
    except:
        result_content = None
    
    extracted = {
        "full_html": final_html,
        "text_content": final_text,
        "result_div_html": result_content,
        "page_url": page.url
    }
    
    log_step(session, "content_extracted", {
        "html_length": len(final_html),
        "text_length": len(final_text),
        "has_result_div": result_content is not None
    })
    
    return extracted
```

**2D: Handle Edge Cases**
```python
async def handle_spa_edge_cases(page, session):
    """Handle common SPA edge cases"""
    
    # Edge Case 1: Loading spinners/overlays
    try:
        spinners = await page.locator(".loading, .spinner, [data-loading='true']").all()
        for spinner in spinners:
            try:
                await spinner.wait_for(state="hidden", timeout=10000)
            except:
                pass
    except:
        pass
    
    # Edge Case 2: Modal dialogs
    try:
        modals = await page.locator(".modal, [role='dialog']").all()
        if modals:
            logger.info(f"Found {len(modals)} modal(s)")
            # Extract modal content
            for modal in modals:
                try:
                    content = await modal.inner_text()
                    if content.strip():
                        logger.info(f"Modal content: {content[:100]}")
                except:
                    pass
    except:
        pass
    
    # Edge Case 3: Shadow DOM
    try:
        # Check for shadow roots
        shadow_content = await page.evaluate("""() => {
            const elements = document.querySelectorAll("*");
            let shadowContent = "";
            elements.forEach(el => {
                if (el.shadowRoot) {
                    shadowContent += el.shadowRoot.innerText + " ";
                }
            });
            return shadowContent;
        }""")
        if shadow_content.strip():
            logger.info(f"Found shadow DOM content: {shadow_content[:100]}")
    except:
        pass
    
    # Edge Case 4: Infinite scroll/lazy loading
    # For text quizzes, just get current viewport content
    await page.evaluate("window.scrollTo(0, 0)")  # Go to top
```

---

### Stage 3: Question Parsing & Decoding (Advanced Multi-Layer)

**3A: Multi-Layer Decoding Function**
```python
def decode_content_multi_layer(content):
    """Handle multiple encoding layers"""
    import base64
    from urllib.parse import unquote
    from html import unescape
    
    decoded_content = content
    layers_decoded = 0
    
    # Try up to 5 layers of base64 encoding
    for attempt in range(5):
        try:
            # Skip if already looks like normal text
            if decoded_content.isprintable() and '\n' in decoded_content:
                break
            
            decoded = base64.b64decode(decoded_content).decode('utf-8')
            if decoded != decoded_content and len(decoded) > 0:
                decoded_content = decoded
                layers_decoded += 1
                logger.debug(f"Base64 decode layer {layers_decoded} successful")
            else:
                break
        except Exception as e:
            logger.debug(f"Base64 decode attempt {attempt + 1} failed: {e}")
            break
    
    # URL decode
    try:
        url_decoded = unquote(decoded_content)
        if url_decoded != decoded_content:
            decoded_content = url_decoded
            logger.debug("URL decode successful")
    except:
        pass
    
    # Handle HTML entities
    try:
        entity_decoded = unescape(decoded_content)
        if entity_decoded != decoded_content:
            decoded_content = entity_decoded
            logger.debug("HTML entity decode successful")
    except:
        pass
    
    return decoded_content, layers_decoded
```

**3B: Structured Question Extraction**
```python
def extract_question_components(question_text, session):
    """Parse question into structured components"""
    import re
    
    components = {
        "question_id": None,
        "question_description": None,
        "data_sources": [],
        "instructions": [],
        "submit_url": None,
        "answer_format": None
    }
    
    # Extract question ID (Q123 format)
    qid_match = re.search(r"Q(\d+)", question_text)
    if qid_match:
        components["question_id"] = f"Q{qid_match.group(1)}"
    
    # Extract URLs (files and endpoints)
    url_pattern = r"https?://[^\s<>\"'\)\]]+(?:\.\w+)?"
    all_urls = re.findall(url_pattern, question_text)
    
    # Categorize URLs
    for url in all_urls:
        if any(ext in url.lower() for ext in ['.pdf', '.csv', '.json', '.xlsx', '.txt']):
            components["data_sources"].append(url)
        elif '/submit' in url.lower() or '/answer' in url.lower():
            components["submit_url"] = url
        else:
            components["data_sources"].append(url)
    
    # Extract special instructions
    instruction_keywords = ['page', 'column', 'sum', 'average', 'filter', 'where', 'group by']
    for keyword in instruction_keywords:
        pattern = rf"(?i){keyword}[\s\S]{{0,50}}"
        matches = re.findall(pattern, question_text)
        components["instructions"].extend(matches[:2])  # Take first 2 matches
    
    # Infer answer format from question
    if any(word in question_text.lower() for word in ['sum', 'total', 'average', 'count', 'percentage']):
        components["answer_format"] = "number"
    elif any(word in question_text.lower() for word in ['true', 'false', 'yes', 'no']):
        components["answer_format"] = "boolean"
    elif any(word in question_text.lower() for word in ['chart', 'image', 'graph', 'plot', 'visualization']):
        components["answer_format"] = "base64_image"
    elif 'json' in question_text.lower():
        components["answer_format"] = "json"
    else:
        components["answer_format"] = "string"
    
    log_step(session, "question_parsed", {
        "question_id": components["question_id"],
        "data_sources_count": len(components["data_sources"]),
        "has_submit_url": components["submit_url"] is not None,
        "inferred_format": components["answer_format"]
    })
    
    return components
```

**3C: LLM-Assisted Question Parsing**
```python
async def parse_question_with_llm(question_text, aipipe_client, session):
    """Use GPT-5-Nano to intelligently parse question"""
    
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
        response = await aipipe_client.create_message(
            model="gpt-5-nano",
            prompt=parse_prompt,
            max_tokens=500,
            timeout=5
        )
        
        parsed = json.loads(response.text)
        
        log_step(session, "llm_question_parsed", {
            "task_type": parsed.get("task_type"),
            "answer_format": parsed.get("answer_format")
        })
        
        return parsed
    
    except Exception as e:
        logger.warning(f"LLM parsing failed: {e}")
        return None
```

---

### Stage 4: Intelligent Task Classification (LLM-Powered)

**4A: Classification with Token Budget Awareness**
```python
async def classify_task(question_text, aipipe_client, gemini_client, token_tracker, session):
    """
    Classify task and select appropriate solving strategy
    Uses GPT-5-Nano unless tokens running low
    
    Now includes scraping method detection for web sources
    """
    
    # Check token budget
    aipipe_usage_pct = token_tracker["aipipe_used"] / token_tracker["aipipe_total"]
    
    # Decide which model to use
    use_model = "gpt5_nano" if aipipe_usage_pct < 0.75 else "gemini"
    
    classify_prompt = f"""Analyze this quiz question and classify:

QUESTION: {question_text}

Return ONLY JSON:
{{
  "task_type": "sourcing|cleansing|analysis|visualization|multi_step",
  "data_formats": ["pdf", "csv", "json", "html", "api", ...],
  "scraping_method": "static|dynamic|hybrid",
  "analysis_required": "aggregation|filtering|statistical|ml|geo|text",
  "complexity": 1-5,
  "expected_answer_type": "number|string|boolean|json|base64_image",
  "estimated_tokens": 300,
  "processing_steps": [
    {{"step": 1, "action": "download", "tool": "requests"}},
    {{"step": 2, "action": "parse", "tool": "pandas"}},
    {{"step": 3, "action": "analyze", "tool": "pandas"}}
  ]
}}"""
    
    try:
        if use_model == "gpt5_nano":
            response = await aipipe_client.create_message(
                model="gpt-5-nano",
                prompt=classify_prompt,
                max_tokens=500,
                timeout=5
            )
            token_tracker["aipipe_used"] += len(classify_prompt) // 4
        else:
            response = await gemini_client.generate_content(
                classify_prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 500}
            )
            token_tracker["gemini_used"] += len(classify_prompt) // 3
        
        classification = json.loads(response.text)
        
        log_step(session, "task_classified", {
            "model_used": use_model,
            "task_type": classification.get("task_type"),
            "scraping_method": classification.get("scraping_method"),
            "complexity": classification.get("complexity")
        })
        
        return classification
    
    except Exception as e:
        logger.error(f"Task classification failed: {e}")
        # Fallback: return generic classification
        return {
            "task_type": "multi_step",
            "scraping_method": "hybrid",
            "complexity": 3,
            "processing_steps": []
        }
```

**4B: URL-Based Scraping Method Detection**
```python
def detect_scraping_method(url: str) -> str:
    """
    Detect whether to use static or dynamic scraping based on URL patterns
    
    Returns: "static" | "dynamic" | "hybrid"
    """
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # Known static sites (fast scraping with requests + BeautifulSoup)
    STATIC_DOMAINS = [
        "wikipedia.org",
        "en.wikipedia.org",
        "*.wikipedia.org",
        "medium.com",
        "github.com",
        "stackoverflow.com",
        "docs.python.org",
        "developer.mozilla.org",
        "w3schools.com",
        "geeksforgeeks.org",
        "tutorialspoint.com",
    ]
    
    # Known dynamic framework sites (require Playwright)
    DYNAMIC_INDICATORS = [
        # Domain patterns
        "vercel.app",
        "netlify.app",
        "herokuapp.com",
        "firebase",
        "amplify",
        # Path patterns suggesting SPA
        "/app/",
        "/dashboard/",
        "/#/",  # Hash routing (Angular/Vue)
    ]
    
    # Check for static domains
    for static_domain in STATIC_DOMAINS:
        if static_domain.startswith("*."):
            if domain.endswith(static_domain[1:]):
                return "static"
        elif static_domain in domain:
            return "static"
    
    # Check for dynamic indicators
    for indicator in DYNAMIC_INDICATORS:
        if indicator in domain or indicator in path:
            return "dynamic"
    
    # File extensions that are always static
    STATIC_EXTENSIONS = ['.html', '.htm', '.txt', '.xml', '.rss']
    if any(path.endswith(ext) for ext in STATIC_EXTENSIONS):
        return "static"
    
    # Default to hybrid (try static first, fallback to dynamic)
    return "hybrid"


# Framework detection patterns for dynamic scraping
FRAMEWORK_SIGNATURES = {
    "vue": {
        "dom_indicators": ["[data-v-", "__VUE__", "data-server-rendered"],
        "script_patterns": ["vue.js", "vue.min.js", "vue.runtime"],
        "attributes": ["v-if", "v-for", "v-bind", "v-model", "@click"]
    },
    "react": {
        "dom_indicators": ["data-reactroot", "_reactRootContainer", "__NEXT_DATA__"],
        "script_patterns": ["react.js", "react.min.js", "react-dom"],
        "attributes": ["data-reactid", "className"]
    },
    "angular": {
        "dom_indicators": ["ng-version", "_nghost", "_ngcontent"],
        "script_patterns": ["angular.js", "angular.min.js", "@angular"],
        "attributes": ["ng-app", "ng-controller", "*ngIf", "*ngFor", "[(ngModel)]"]
    },
    "svelte": {
        "dom_indicators": ["svelte-"],
        "script_patterns": ["svelte"],
        "attributes": []
    }
}
```

---

### Stage 5: Data Sourcing Module (Hybrid Web Scraping)

**5A: Multi-Source Data Fetching with Hybrid Scraping**
```python
import random
import asyncio
from bs4 import BeautifulSoup
import html2text

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
    """Get a random User-Agent for request rotation"""
    return random.choice(USER_AGENTS)


async def fetch_all_data_sources(data_sources, session, page=None):
    """
    Fetch data from all identified sources using hybrid scraping strategy
    
    Args:
        data_sources: List of URLs to fetch
        session: Pipeline session state
        page: Optional Playwright page for dynamic scraping
    """
    
    fetched_data = {}
    
    for source_url in data_sources:
        try:
            # Determine source type by extension
            if source_url.endswith('.pdf'):
                fetched_data[source_url] = await download_and_parse_pdf(source_url)
            elif source_url.endswith(('.csv', '.json', '.xlsx')):
                fetched_data[source_url] = await download_and_parse_file(source_url)
            elif '/api/' in source_url or source_url.endswith('/data'):
                fetched_data[source_url] = await call_api(source_url)
            else:
                # Use hybrid web scraping for HTML pages
                fetched_data[source_url] = await scrape_webpage_hybrid(
                    source_url, session, page
                )
            
            log_step(session, "data_fetched", {
                "source": source_url,
                "status": "success",
                "method": fetched_data[source_url].get("_scrape_method", "unknown")
            })
        
        except Exception as e:
            logger.error(f"Failed to fetch {source_url}: {e}")
            log_step(session, "data_fetch_failed", {
                "source": source_url,
                "error": str(e)
            })
    
    return fetched_data
```

**5B: Hybrid Web Scraping - Smart Method Selection**
```python
async def scrape_webpage_hybrid(url: str, session, page=None) -> dict:
    """
    Smart hybrid scraping that selects the optimal method:
    
    1. Detect if static or dynamic scraping is needed
    2. Try fast static scraping first (requests + BeautifulSoup)
    3. Auto-switch to Playwright if content is JS-rendered
    4. Extract structured data (tables, lists, key-value pairs)
    
    Returns:
        dict with keys: html, text, tables, lists, key_values, _scrape_method
    """
    
    scrape_method = detect_scraping_method(url)
    logger.info(f"Scraping {url} with method: {scrape_method}")
    
    result = {
        "html": "",
        "text": "",
        "tables": [],
        "lists": [],
        "key_values": {},
        "_scrape_method": scrape_method,
        "_url": url
    }
    
    if scrape_method == "static" or scrape_method == "hybrid":
        # Try fast static scraping first
        static_result = await scrape_static(url)
        
        if static_result and static_result.get("_valid"):
            result.update(static_result)
            result["_scrape_method"] = "static"
            return result
        
        # If static failed or content seems JS-rendered, switch to dynamic
        if scrape_method == "hybrid":
            logger.info(f"Static scraping insufficient for {url}, switching to dynamic")
    
    # Dynamic scraping with Playwright
    if page:
        dynamic_result = await scrape_dynamic(url, page, session)
        result.update(dynamic_result)
        result["_scrape_method"] = "dynamic"
    else:
        logger.warning(f"No Playwright page available for dynamic scraping of {url}")
    
    return result
```

**5C: Method A - Fast Static Scraping (Wikipedia/Blogs)**
```python
async def scrape_static(url: str) -> dict:
    """
    Fast static scraping using requests + BeautifulSoup
    
    Best for: Wikipedia, news articles, blogs, simple HTML pages
    Speed: ~100-500ms per page
    """
    import aiohttp
    from bs4 import BeautifulSoup
    
    result = {
        "html": "",
        "text": "",
        "tables": [],
        "lists": [],
        "key_values": {},
        "_valid": False
    }
    
    try:
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    logger.warning(f"Static scrape got status {resp.status}")
                    return result
                
                html = await resp.text()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script and style elements
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
            tag.decompose()
        
        # Check if content is JS-rendered (empty divs, script-heavy)
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text_content = main_content.get_text(strip=True)
            
            # Heuristic: if very little text but many script tags, likely JS-rendered
            script_count = len(soup.find_all('script'))
            if len(text_content) < 100 and script_count > 5:
                logger.info("Page appears to be JS-rendered, static scraping insufficient")
                return result
        
        result["html"] = str(soup)
        result["text"] = soup.get_text(separator='\n', strip=True)
        
        # Extract structured data
        result["tables"] = extract_html_tables_to_df(soup)
        result["lists"] = extract_list_items(soup)
        result["key_values"] = extract_key_value_pairs(soup)
        
        # Mark as valid if we got meaningful content
        result["_valid"] = len(result["text"]) > 50
        
        return result
    
    except Exception as e:
        logger.error(f"Static scraping failed: {e}")
        return result
```

**5D: Method B - Dynamic Framework Scraping (Vue/React/Angular)**
```python
async def scrape_dynamic(url: str, page, session) -> dict:
    """
    Dynamic scraping using Playwright for JS-rendered pages
    
    Handles:
    - Vue.js, React, Angular, Svelte frameworks
    - Pagination (Next buttons)
    - Click-to-reveal content
    - Infinite scroll
    
    Speed: ~2-10s per page (includes rendering time)
    """
    import re
    
    result = {
        "html": "",
        "text": "",
        "tables": [],
        "lists": [],
        "key_values": {},
        "pages_scraped": 1
    }
    
    try:
        # Navigate with anti-detection
        await page.goto(url, wait_until="networkidle", timeout=30000)
        
        # Wait for framework-specific indicators
        detected_framework = await detect_and_wait_for_framework(page)
        logger.info(f"Detected framework: {detected_framework or 'unknown'}")
        
        # Handle loading states
        await wait_for_content_load(page)
        
        # Handle click-to-reveal content
        await handle_click_to_reveal(page)
        
        # Collect initial page content
        all_html = [await page.content()]
        all_text = [await page.evaluate("document.body.innerText")]
        
        # Handle pagination if present
        pagination_result = await handle_pagination(page, session)
        if pagination_result:
            all_html.extend(pagination_result["html_pages"])
            all_text.extend(pagination_result["text_pages"])
            result["pages_scraped"] = pagination_result["page_count"]
        
        # Handle infinite scroll if no pagination found
        elif await detect_infinite_scroll(page):
            scroll_result = await handle_infinite_scroll(page)
            all_html = [scroll_result["html"]]
            all_text = [scroll_result["text"]]
        
        # Combine all content
        result["html"] = "\n".join(all_html)
        result["text"] = "\n".join(all_text)
        
        # Parse combined HTML for structured data
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(result["html"], 'lxml')
        result["tables"] = extract_html_tables_to_df(soup)
        result["lists"] = extract_list_items(soup)
        result["key_values"] = extract_key_value_pairs(soup)
        
        return result
    
    except Exception as e:
        logger.error(f"Dynamic scraping failed: {e}")
        return result


async def detect_and_wait_for_framework(page) -> str:
    """Detect JS framework and wait for its specific render indicators"""
    
    # Vue.js detection
    try:
        vue_detected = await page.evaluate("""() => {
            return !!(window.__VUE__ || 
                     window.__VUE_DEVTOOLS_GLOBAL_HOOK__ ||
                     document.querySelector('[data-v-]') ||
                     document.querySelector('[data-server-rendered]'))
        }""")
        if vue_detected:
            # Wait for Vue-specific attributes
            await page.wait_for_selector('[data-v-]', timeout=5000)
            return "vue"
    except:
        pass
    
    # React detection
    try:
        react_detected = await page.evaluate("""() => {
            return !!(window.__REACT_DEVTOOLS_GLOBAL_HOOK__ ||
                     document.querySelector('[data-reactroot]') ||
                     document.querySelector('#__next') ||
                     document.querySelector('#root')?._reactRootContainer)
        }""")
        if react_detected:
            await page.wait_for_selector('[data-reactroot], #__next, #root', timeout=5000)
            return "react"
    except:
        pass
    
    # Angular detection
    try:
        angular_detected = await page.evaluate("""() => {
            return !!(window.ng || 
                     document.querySelector('[ng-version]') ||
                     document.querySelector('[_nghost]'))
        }""")
        if angular_detected:
            await page.wait_for_selector('[ng-version], [_nghost]', timeout=5000)
            return "angular"
    except:
        pass
    
    # Generic wait for content
    await page.wait_for_load_state("networkidle")
    return None


async def wait_for_content_load(page):
    """Wait for dynamic content to fully load"""
    
    # Wait for loading spinners to disappear
    spinners = [
        ".loading", ".spinner", ".loader",
        "[data-loading='true']", "[aria-busy='true']",
        ".skeleton", ".placeholder"
    ]
    
    for spinner_selector in spinners:
        try:
            spinner = page.locator(spinner_selector)
            if await spinner.count() > 0:
                await spinner.first.wait_for(state="hidden", timeout=10000)
        except:
            pass
    
    # Wait for main content area to have content
    try:
        await page.wait_for_function("""() => {
            const main = document.querySelector('main, article, .content, #content, .main');
            return main && main.innerText.trim().length > 100;
        }""", timeout=10000)
    except:
        pass


async def handle_click_to_reveal(page):
    """Click elements that reveal hidden content"""
    
    reveal_selectors = [
        "button:has-text('Show More')",
        "button:has-text('View More')",
        "button:has-text('Load More')",
        "button:has-text('See More')",
        "button:has-text('Expand')",
        "a:has-text('Show More')",
        "[data-toggle='collapse']",
        ".show-more",
        ".view-more",
        ".expand-btn"
    ]
    
    for selector in reveal_selectors:
        try:
            elements = await page.locator(selector).all()
            for element in elements[:5]:  # Limit to 5 clicks
                if await element.is_visible():
                    await element.click()
                    await asyncio.sleep(random.uniform(0.3, 0.8))
        except:
            pass


async def handle_pagination(page, session, max_pages: int = 10) -> dict:
    """
    Handle pagination by clicking Next buttons
    
    Detects common pagination patterns:
    - "Next" / ">" / "»" buttons
    - Page number links
    - "Load More" buttons
    """
    import re
    
    result = {
        "html_pages": [],
        "text_pages": [],
        "page_count": 1
    }
    
    # Pagination button patterns (regex)
    next_patterns = [
        r'^>$', r'^>>$', r'^»$', r'^›$',
        r'^Next$', r'^Next\s*Page$', r'^Next\s*→$',
        r'^\s*>\s*$'
    ]
    
    next_selectors = [
        "a:has-text('Next')",
        "button:has-text('Next')",
        "a[rel='next']",
        ".pagination .next",
        ".pager .next",
        "[aria-label='Next']",
        "[aria-label='Next page']",
        ".next-page",
        "a.next",
        "button.next"
    ]
    
    for page_num in range(2, max_pages + 1):
        # Check time budget
        if time_remaining_safe(session) < 30:
            logger.warning("Time budget low, stopping pagination")
            break
        
        clicked = False
        
        # Try each selector
        for selector in next_selectors:
            try:
                next_btn = page.locator(selector).first
                if await next_btn.is_visible() and await next_btn.is_enabled():
                    # Random delay for anti-blocking
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                    
                    await next_btn.click()
                    await page.wait_for_load_state("networkidle")
                    await wait_for_content_load(page)
                    
                    result["html_pages"].append(await page.content())
                    result["text_pages"].append(await page.evaluate("document.body.innerText"))
                    result["page_count"] = page_num
                    
                    clicked = True
                    logger.info(f"Pagination: scraped page {page_num}")
                    break
            except:
                continue
        
        if not clicked:
            # No more pages or pagination not found
            break
    
    return result if result["page_count"] > 1 else None


async def detect_infinite_scroll(page) -> bool:
    """Detect if page uses infinite scroll instead of pagination"""
    
    try:
        # Check for common infinite scroll indicators
        has_infinite_scroll = await page.evaluate("""() => {
            // Check for intersection observers or scroll listeners
            const hasObserver = !!window.IntersectionObserver;
            
            // Check for "load more" sentinel elements
            const sentinel = document.querySelector(
                '.infinite-scroll-component, [data-infinite], .load-more-trigger'
            );
            
            // Check if scrolling loads new content
            const scrollHeight = document.body.scrollHeight;
            
            return hasObserver && scrollHeight > window.innerHeight * 2;
        }""")
        
        return has_infinite_scroll
    except:
        return False


async def handle_infinite_scroll(page, max_scrolls: int = 10) -> dict:
    """Handle infinite scroll pages by scrolling to load content"""
    
    result = {"html": "", "text": ""}
    
    previous_height = 0
    scroll_count = 0
    
    while scroll_count < max_scrolls:
        # Scroll to bottom
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(random.uniform(1.0, 2.0))
        
        # Check if new content loaded
        new_height = await page.evaluate("document.body.scrollHeight")
        
        if new_height == previous_height:
            # No new content, stop scrolling
            break
        
        previous_height = new_height
        scroll_count += 1
        logger.debug(f"Infinite scroll: scroll {scroll_count}, height {new_height}")
    
    # Scroll back to top and get full content
    await page.evaluate("window.scrollTo(0, 0)")
    result["html"] = await page.content()
    result["text"] = await page.evaluate("document.body.innerText")
    
    return result
```

**5E: LLM-Guided Selector Generation**
```python
async def generate_css_selector_with_llm(
    html_snippet: str, 
    target_entity: str,
    llm_client,
    session
) -> dict:
    """
    Use LLM to generate CSS selectors for target data extraction
    
    Args:
        html_snippet: Sample HTML from the page (first 2000 chars)
        target_entity: What we're looking for (e.g., "product prices", "table rows")
        llm_client: Aipipe or Gemini client
        session: Pipeline session
    
    Returns:
        dict with: selector, fallback_selector, visual_parsing (bool)
    """
    
    # Truncate HTML for token efficiency
    html_sample = html_snippet[:2000] if len(html_snippet) > 2000 else html_snippet
    
    selector_prompt = f"""Analyze this HTML snippet and return CSS selectors for extracting: {target_entity}

HTML SNIPPET:
{html_sample}

Return ONLY valid JSON:
{{
  "primary_selector": "<CSS selector for main target>",
  "fallback_selector": "<alternative CSS selector>",
  "data_type": "table|list|text|key_value",
  "visual_parsing": false,
  "notes": "<brief extraction notes>"
}}

If the structure is too complex or dynamic for CSS selectors, set "visual_parsing": true.
"""
    
    try:
        response = await llm_client.generate(
            selector_prompt,
            max_tokens=300,
            temperature=0.1
        )
        
        result = json.loads(response)
        
        log_step(session, "selector_generated", {
            "target": target_entity,
            "selector": result.get("primary_selector"),
            "visual_parsing": result.get("visual_parsing", False)
        })
        
        return result
    
    except Exception as e:
        logger.warning(f"LLM selector generation failed: {e}")
        return {"visual_parsing": True}


async def extract_with_visual_parsing(page, target_entity: str, llm_client) -> str:
    """
    Fallback: Convert page to Markdown and use LLM to extract data
    
    Used when CSS selectors fail or page structure is too complex
    """
    
    # Get page HTML
    html = await page.content()
    
    # Convert to Markdown using html2text
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.ignore_images = True
    h2t.ignore_tables = False
    markdown_content = h2t.handle(html)
    
    # Truncate for token efficiency
    markdown_sample = markdown_content[:4000] if len(markdown_content) > 4000 else markdown_content
    
    extraction_prompt = f"""Extract {target_entity} from this page content (Markdown format):

{markdown_sample}

Return the extracted data as JSON. If it's tabular data, return as array of objects.
If it's a single value, return {{"value": "<extracted value>"}}.
"""
    
    try:
        response = await llm_client.generate(
            extraction_prompt,
            max_tokens=1000,
            temperature=0.1
        )
        
        return json.loads(response)
    
    except Exception as e:
        logger.error(f"Visual parsing failed: {e}")
        return None
```

**5F: Specialized Extraction Helpers**
```python
def extract_html_tables_to_df(soup) -> list:
    """
    Robust HTML table extraction with colspan/rowspan handling
    
    Returns list of pandas DataFrames
    """
    import pandas as pd
    from bs4 import BeautifulSoup
    
    tables = []
    
    for table in soup.find_all('table'):
        try:
            # Try pandas read_html first (handles most cases)
            html_str = str(table)
            dfs = pd.read_html(html_str)
            if dfs:
                tables.extend(dfs)
                continue
        except:
            pass
        
        # Manual parsing for complex tables with colspan/rowspan
        try:
            rows = table.find_all('tr')
            if not rows:
                continue
            
            # Build matrix accounting for colspan/rowspan
            matrix = []
            rowspan_tracker = {}  # {col_idx: (value, remaining_rows)}
            
            for row_idx, row in enumerate(rows):
                cells = row.find_all(['th', 'td'])
                matrix_row = []
                col_idx = 0
                cell_idx = 0
                
                while col_idx < 100:  # Safety limit
                    # Check if there's a rowspan cell to fill
                    if col_idx in rowspan_tracker:
                        value, remaining = rowspan_tracker[col_idx]
                        matrix_row.append(value)
                        if remaining > 1:
                            rowspan_tracker[col_idx] = (value, remaining - 1)
                        else:
                            del rowspan_tracker[col_idx]
                        col_idx += 1
                        continue
                    
                    if cell_idx >= len(cells):
                        break
                    
                    cell = cells[cell_idx]
                    cell_text = cell.get_text(strip=True)
                    colspan = int(cell.get('colspan', 1))
                    rowspan = int(cell.get('rowspan', 1))
                    
                    # Handle colspan
                    for _ in range(colspan):
                        matrix_row.append(cell_text)
                        
                        # Track rowspan for future rows
                        if rowspan > 1:
                            rowspan_tracker[col_idx] = (cell_text, rowspan - 1)
                        
                        col_idx += 1
                    
                    cell_idx += 1
                
                matrix.append(matrix_row)
            
            # Convert to DataFrame
            if matrix:
                # Use first row as header if it looks like headers
                if matrix[0] and all(isinstance(x, str) for x in matrix[0]):
                    df = pd.DataFrame(matrix[1:], columns=matrix[0])
                else:
                    df = pd.DataFrame(matrix)
                tables.append(df)
        
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
    
    return tables


def extract_list_items(soup) -> list:
    """
    Extract <ul>/<ol> lists to Python lists
    
    Returns list of dicts with type and items
    """
    from bs4 import BeautifulSoup
    
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


def extract_key_value_pairs(soup) -> dict:
    """
    Extract key-value pairs from various HTML patterns:
    - Definition lists <dl><dt><dd>
    - Label: Value text patterns
    - Table rows with 2 columns
    """
    from bs4 import BeautifulSoup
    import re
    
    pairs = {}
    
    # Pattern 1: Definition lists
    for dl in soup.find_all('dl'):
        dts = dl.find_all('dt')
        dds = dl.find_all('dd')
        for dt, dd in zip(dts, dds):
            key = dt.get_text(strip=True)
            value = dd.get_text(strip=True)
            if key and value:
                pairs[key] = value
    
    # Pattern 2: Label: Value in text (e.g., "Price: $99.99")
    text = soup.get_text()
    kv_pattern = r'([A-Za-z][A-Za-z\s]{1,30}):\s*([^\n:]{1,100})'
    matches = re.findall(kv_pattern, text)
    for key, value in matches:
        key = key.strip()
        value = value.strip()
        if key and value and len(key) < 50:
            pairs[key] = value
    
    # Pattern 3: Two-column tables (key in col 1, value in col 2)
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                if key and value and len(key) < 50:
                    pairs[key] = value
    
    return pairs
```

**5G: PDF and File Parsing (Enhanced)**
```python
async def download_and_parse_pdf(url):
    """Download and extract content from PDF"""
    import pypdf
    import pdfplumber
    import io
    
    # Download file with User-Agent rotation
    file_bytes = await download_file(url)
    
    # Extract with pdfplumber
    extracted = {
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
    
    return extracted


async def download_file(url, max_retries=3):
    """Download file with retry logic and User-Agent rotation"""
    import aiohttp
    
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": get_random_user_agent(),
                "Accept": "*/*",
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=False) as resp:
                    if resp.status == 200:
                        return await resp.read()
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception(f"Failed to download {url} after {max_retries} attempts")
```

---

### Stage 6: Data Cleansing & Transformation

**6A: Smart Data Cleaning**
```python
def clean_and_prepare_data(raw_data, question_context):
    """Convert raw data into clean pandas DataFrame"""
    import pandas as pd
    
    # Handle different input types
    if isinstance(raw_data, str):
        # Try parsing as JSON first
        try:
            df = pd.read_json(raw_data)
        except:
            # Try parsing as CSV
            try:
                df = pd.read_csv(raw_data)
            except:
                # Last resort: split by common delimiters
                df = pd.DataFrame([raw_data.split()])
    
    elif isinstance(raw_data, dict):
        df = pd.DataFrame(raw_data)
    
    elif isinstance(raw_data, list):
        df = pd.DataFrame(raw_data)
    
    else:
        df = pd.DataFrame(raw_data)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
    
    # Infer and convert column types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass  # Keep as string
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('UNKNOWN')
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    return df
```

---

### Stage 7: Data Analysis Module (LLM-Powered)

**7A: Intelligent Analysis Planning**
```python
async def plan_analysis_with_llm(df, question_text, aipipe_client, gemini_client, token_tracker):
    """Generate analysis code using LLM"""
    
    # Show LLM the data structure
    sample_data = df.head(5).to_string()
    schema = str(df.dtypes.to_dict())
    
    analysis_prompt = f"""You are a pandas expert. Given:

QUESTION: {question_text}
DATAFRAME SHAPE: {df.shape}
COLUMNS: {list(df.columns)}
DTYPES: {schema}
SAMPLE DATA:
{sample_data}

Generate ONLY Python pandas code (NO explanations):
- Use variable 'df' for the dataframe
- Store final answer in variable 'answer'
- For numeric: ensure correct precision
- For aggregations: preserve structure if needed

Code:
"""
    
    # Select model based on token budget
    aipipe_usage_pct = token_tracker["aipipe_used"] / token_tracker["aipipe_total"]
    
    try:
        if aipipe_usage_pct < 0.75:
            response = await aipipe_client.create_message(
                model="gpt-5-nano",
                prompt=analysis_prompt,
                max_tokens=1000,
                timeout=5
            )
        else:
            response = await gemini_client.generate_content(
                analysis_prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 1000}
            )
        
        return response.text
    except Exception as e:
        logger.error(f"Analysis planning failed: {e}")
        return None
```

**7B: Sandboxed Code Execution**
```python
async def execute_analysis_code(code, df):
    """Execute generated pandas code safely"""
    import sys
    from io import StringIO
    
    # Create sandbox namespace
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "answer": None
    }
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = output_buffer = StringIO()
    
    try:
        exec(code, namespace)
        answer = namespace.get("answer")
        output = output_buffer.getvalue()
        
        return answer, output, None
    
    except Exception as e:
        return None, "", str(e)
    
    finally:
        sys.stdout = old_stdout
```

---

### Stage 8-10: Visualization, Formatting & Submission

*(Continue with Stages 8-10 as defined in previous version)*

---

### Stage 11: Error Handling & Timeout Management

**11A: Strict Deadline Enforcement**
```python
def time_remaining_safe(session):
    """Get safe time remaining before deadline"""
    elapsed = time.time() - session["start_time"]
    remaining = session["deadline"] - time.time()
    
    SAFETY_BUFFER = 10  # Reserve 10 seconds for final submission
    safe_remaining = remaining - SAFETY_BUFFER
    
    logger.debug(f"Time: elapsed={elapsed:.1f}s, remaining={remaining:.1f}s, safe={safe_remaining:.1f}s")
    
    return max(0, safe_remaining)

async def force_submit_if_timeout(session, submit_func):
    """Submit current best answer if approaching deadline"""
    if time_remaining_safe(session) < 15:
        logger.warning(f"⏰ TIME CRITICAL: Only {time_remaining_safe(session):.0f}s left!")
        
        if session["final_answer"] is None:
            session["final_answer"] = "TIMEOUT_NO_ANSWER"
        
        # Force submit immediately
        return await submit_func(session)
```

---

### Stage 12: Main Quiz Pipeline (Orchestration)

**12A: Complete Quiz Solving Pipeline**
```python
async def solve_quiz_pipeline(email, secret, initial_url, deadline):
    """
    Main pipeline orchestrating the entire quiz solving process
    Handles multiple chained quizzes within 3-minute deadline
    """
    
    # Initialize
    session = await create_session_from_request(email, secret, initial_url, deadline)
    playwright, browser, context, page = await launch_browser()
    aipipe_client, gemini_client, token_tracker = await initialize_llm_clients()
    
    try:
        current_url = initial_url
        quiz_count = 0
        MAX_QUIZZES = 10  # Safety limit
        
        while current_url and quiz_count < MAX_QUIZZES and time_remaining_safe(session) > 30:
            
            quiz_count += 1
            logger.info(f"\n{'='*60}\n📋 QUIZ {quiz_count}: {current_url}\n{'='*60}")
            
            # Stage 2: Navigate & render
            await navigate_to_quiz(page, current_url, session)
            await wait_for_spa_rendering(page, session)
            
            # Extract content
            extracted = await extract_and_decode_content(page, session)
            raw_question, layers = decode_content_multi_layer(
                extracted["result_div_html"] or extracted["text_content"]
            )
            
            # Stage 3: Parse question
            question_components = extract_question_components(raw_question, session)
            session["question_id"] = question_components["question_id"]
            session["submit_url"] = question_components["submit_url"]
            session["expected_format"] = question_components["answer_format"]
            
            # Stage 4: Classify task
            classification = await classify_task(
                raw_question, aipipe_client, gemini_client, token_tracker, session
            )
            
            # Stage 5: Fetch data
            if question_components["data_sources"]:
                fetched_data = await fetch_all_data_sources(
                    question_components["data_sources"], session
                )
                session["raw_data"] = fetched_data
            
            # Stage 6: Clean data
            if session["raw_data"]:
                session["cleaned_data"] = clean_and_prepare_data(
                    session["raw_data"], raw_question
                )
            
            # Stage 7: Analyze
            if session["cleaned_data"] is not None:
                analysis_code = await plan_analysis_with_llm(
                    session["cleaned_data"], raw_question,
                    aipipe_client, gemini_client, token_tracker
                )
                answer, output, error = await execute_analysis_code(
                    analysis_code, session["cleaned_data"]
                )
                session["analysis_result"] = answer
            
            # Stage 9: Format answer
            session["final_answer"] = format_answer(
                session["analysis_result"],
                session["expected_format"]
            )
            
            # Stage 10: Submit answer
            logger.info(f"Submitting answer: {session['final_answer']}")
            submission_response = await submit_answer(
                email, secret, current_url,
                session["final_answer"],
                session["submit_url"]
            )
            
            # Process response
            if submission_response.get("correct"):
                logger.info("✅ CORRECT!")
                if submission_response.get("url"):
                    current_url = submission_response["url"]
                    session["quiz_chain"].append(current_url)
                else:
                    logger.info("🎉 QUIZ COMPLETE!")
                    break
            
            else:
                logger.warning(f"❌ INCORRECT: {submission_response.get('reason')}")
                
                if submission_response.get("url"):
                    logger.info(f"Moving to new quiz: {submission_response['url']}")
                    current_url = submission_response["url"]
                else:
                    logger.info("Retrying current quiz...")
                    # Retry logic here
        
        logger.info(f"\n{'='*60}\n✅ PIPELINE COMPLETE\n{'='*60}")
        log_step(session, "pipeline_complete", {
            "quizzes_solved": quiz_count,
            "total_time": time.time() - session["start_time"]
        })
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        log_step(session, "pipeline_error", {"error": str(e)})
    
    finally:
        await browser.close()
        await playwright.stop()
```

---

## Deployment & Testing

### Test Your Endpoint
```bash
# Test endpoint with demo
curl -X POST https://your-domain.com/api/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your_secret",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'

# Expected response (HTTP 200):
# {
#   "status": "accepted",
#   "quiz_id": "quiz_...",
#   "timestamp": "2025-11-28T..."
# }
```

### Error Handling Test Cases
```bash
# Test invalid JSON (expect HTTP 400)
curl -X POST https://your-domain.com/api/quiz \
  -H "Content-Type: application/json" \
  -d 'invalid json'

# Test invalid secret (expect HTTP 403)
curl -X POST https://your-domain.com/api/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "secret": "wrong_secret",
    "url": "https://example.com"
  }'
```

---

## Key Success Points

✅ **HTTP Error Handling**:
- 200 OK: Valid request (secret verified)
- 400 Bad Request: Malformed JSON
- 403 Forbidden: Invalid secret

✅ **3-Minute Deadline**:
- Timer starts from POST reception
- 10-second safety buffer before final submission
- Force submit if timeout approaching

✅ **Chained Quizzes**:
- Follow new URLs until quiz complete
- Respect overall 3-minute deadline across all quizzes
- Track audit trail for debugging

✅ **Production Ready**:
- Async/await for concurrency
- Comprehensive error logging
- Token budget tracking
- Vue.js/React/Angular support
- Dynamic URL extraction (no hardcoding)

---

**Remember**: Your API endpoint must validate requests quickly, offload quiz solving to async tasks, and ensure all submissions happen within the 3-minute window. All URLs are extracted dynamically from quiz pages - never hardcode endpoints.
