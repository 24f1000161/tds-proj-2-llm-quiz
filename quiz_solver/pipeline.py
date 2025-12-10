"""
Main quiz solving pipeline orchestration.
"""

import time
import asyncio
import hashlib
from typing import Any, Optional
from urllib.parse import urljoin
import pandas as pd

from .config import settings
from .models import SessionState, AnswerFormat
from .browser import (
    BrowserManager,
    navigate_to_quiz,
    wait_for_spa_rendering,
    extract_and_decode_content,
    decode_content_multi_layer,
    handle_spa_edge_cases
)
from .llm_client import llm_client
from .data_sourcing import (
    fetch_all_data_sources,
    clean_and_prepare_data,
    get_dataframe_info
)
from .question_parser import extract_question_components, format_answer
from .analysis import execute_analysis_code, simple_analysis
from .submission import submit_answer
from .logging_utils import logger, log_step


def generate_quiz_id(url: str) -> str:
    """Generate a unique quiz ID from URL."""
    return hashlib.md5(f"{url}-{time.time()}".encode()).hexdigest()[:12]


def create_session(email: str, secret: str, initial_url: str, deadline: float) -> SessionState:
    """Create a new session state for quiz solving."""
    
    return SessionState(
        email=email,
        secret=secret,
        initial_url=initial_url,
        start_time=time.time(),
        deadline=deadline,
        current_url=initial_url,
        quiz_chain=[initial_url]
    )


def time_remaining_safe(session: SessionState) -> float:
    """Get safe time remaining before deadline."""
    remaining = session.deadline - time.time()
    safe_remaining = remaining - session.deadline_safety_buffer
    return max(0, safe_remaining)


def should_force_submit(session: SessionState) -> bool:
    """Check if we should force submit due to time pressure."""
    return time_remaining_safe(session) <= 5


# extract_answer_from_context removed - using LLM-driven approach instead


async def scrape_url_with_browser(page, url: str, session: SessionState) -> dict:
    """Scrape a URL using the browser and return both text and HTML content."""
    logger.info(f"🌐 BROWSER SCRAPE: Navigating to {url}")
    
    result = {"text": "", "html": ""}
    
    try:
        # Navigate to the URL
        response = await page.goto(url, wait_until="networkidle", timeout=15000)
        logger.info(f"📄 Response status: {response.status if response else 'unknown'}")
        
        # Wait for JavaScript to render (reduced wait)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(0.5)  # Short wait for dynamic content
        
        # Get full HTML for CSS selector scraping
        result["html"] = await page.content()
        
        # Try to get content from common containers
        content = ""
        
        # Try #question or #result divs first
        for selector in ["#question", "#result", "#content", "main", "article", "body"]:
            try:
                element = page.locator(selector).first
                if await element.count() > 0:
                    text = await element.inner_text()
                    if text and len(text.strip()) > 10:
                        content = text.strip()
                        logger.info(f"📝 Found content in '{selector}': {len(content)} chars")
                        break
            except Exception:
                continue
        
        if not content:
            # Fallback to full page text
            content = await page.evaluate("document.body.innerText")
            logger.info(f"📝 Full page content: {len(content)} chars")
        
        result["text"] = content
        logger.info(f"📝 Scraped content preview: {content[:200]}...")
        return result
        
    except Exception as e:
        logger.error(f"❌ Browser scrape failed: {e}")
        return ""


async def solve_quiz_pipeline(
    email: str,
    secret: str,
    initial_url: str,
    deadline: float
) -> None:
    """
    Main pipeline orchestrating the entire quiz solving process.
    Handles multiple chained quizzes within 3-minute deadline.
    """
    
    # Initialize session
    session = create_session(email, secret, initial_url, deadline)
    session.quiz_id = generate_quiz_id(initial_url)
    
    logger.info("=" * 70)
    logger.info("🚀 QUIZ SOLVER PIPELINE STARTED")
    logger.info(f"   Email: {email}")
    logger.info(f"   Initial URL: {initial_url}")
    logger.info(f"   Deadline: {settings.timeouts.quiz_deadline_seconds}s")
    logger.info("=" * 70)
    
    # Initialize LLM client
    logger.info("🤖 Initializing LLM client...")
    await llm_client.initialize()
    logger.info("   ✓ LLM client ready")
    
    # Initialize browser
    logger.info("🌐 Launching browser...")
    browser_manager = BrowserManager()
    
    try:
        page = await browser_manager.launch()
        logger.info("   ✓ Browser launched")
        
        current_url = initial_url
        quiz_count = 0
        MAX_QUIZZES = 10  # Safety limit
        
        while current_url and quiz_count < MAX_QUIZZES:
            
            quiz_count += 1
            
            # Reset deadline for each quiz - 3 minutes per question
            session.start_time = time.time()
            session.deadline = session.start_time + settings.timeouts.quiz_deadline_seconds
            
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"📋 QUIZ {quiz_count}: {current_url}")
            logger.info(f"⏱️  New 3-minute deadline started")
            logger.info("=" * 70)
            
            try:
                # ==================== STAGE 1: NAVIGATE ====================
                logger.info("")
                logger.info("📍 STAGE 1: Navigation")
                await navigate_to_quiz(page, current_url, session)
                logger.info("   ✓ Page loaded")
                
                # ==================== STAGE 2: RENDER & EXTRACT ====================
                logger.info("")
                logger.info("🔍 STAGE 2: Content Extraction")
                await wait_for_spa_rendering(page, session)
                await handle_spa_edge_cases(page, session)
                
                extracted = await extract_and_decode_content(page, session)
                raw_question, layers = decode_content_multi_layer(
                    extracted["result_div_html"] or extracted["text_content"]
                )
                
                session.question_text = raw_question
                logger.info(f"   ✓ Extracted question ({len(raw_question)} chars)")
                logger.info(f"   📝 Question preview:")
                for line in raw_question[:500].split('\n')[:10]:
                    if line.strip():
                        logger.info(f"      {line.strip()}")
                
                # Log extracted links and special values
                if extracted.get("links"):
                    logger.info(f"   📎 Found {len(extracted['links'])} links on page:")
                    for link in extracted['links']:
                        logger.info(f"      - {link.get('href')} ({link.get('text', 'no text')})")
                if extracted.get("media_sources"):
                    logger.info(f"   🎵 Media sources: {extracted['media_sources']}")
                if extracted.get("special_values"):
                    logger.info(f"   📊 Special values: {extracted['special_values']}")
                
                # ==================== STAGE 3: PARSE QUESTION ====================
                logger.info("")
                logger.info("🧩 STAGE 3: Question Parsing")
                question_components = extract_question_components(raw_question, session)
                
                # Keep all discovered data sources, even template URLs with placeholders
                # We may replace placeholders later using auxiliary config files (e.g., gh-tree.json)
                
                session.question_id = question_components.question_id
                session.submit_url = question_components.submit_url
                session.expected_format = question_components.answer_format
                session.data_sources = question_components.data_sources
                
                logger.info(f"   • Question ID: {question_components.question_id or 'N/A'}")
                logger.info(f"   • Submit URL: {question_components.submit_url or 'default'}")
                logger.info(f"   • Answer format: {question_components.answer_format}")
                logger.info(f"   • Data sources found: {len(question_components.data_sources)}")
                for ds in question_components.data_sources[:3]:
                    logger.info(f"     - {ds}")
                if question_components.relative_urls:
                    logger.info(f"   • Relative URLs to scrape: {question_components.relative_urls}")
                if question_components.relative_submit_url:
                    logger.info(f"   • Relative submit URL: {question_components.relative_submit_url}")
                
                # ==================== STAGE 4: LLM CLASSIFICATION ====================
                logger.info("")
                logger.info("🤖 STAGE 4: LLM Task Classification")
                classification = {}
                
                try:
                    classification = await llm_client.classify_task(raw_question)
                    logger.info(f"   • Task type: {classification.get('task_type', 'unknown')}")
                    logger.info(f"   • Complexity: {classification.get('complexity', 'unknown')}")
                    logger.info(f"   • Answer type: {classification.get('expected_answer_type', 'unknown')}")
                    
                    # Check if LLM identifies URLs to scrape that we missed
                    llm_data_sources = classification.get('data_sources', [])
                    if llm_data_sources:
                        logger.info(f"   • LLM identified sources: {llm_data_sources}")
                        for src in llm_data_sources:
                            if src not in question_components.data_sources:
                                question_components.data_sources.append(src)
                except Exception as e:
                    logger.warning(f"   ⚠️ LLM classification failed: {e}")
                    logger.info("   → Continuing with rule-based processing...")
                
                # ==================== STAGE 5: DATA SOURCING ====================
                logger.info("")
                logger.info("📥 STAGE 5: Data Sourcing")
                fetched_data: dict[str, Any] = {}
                merged_context: dict[str, Any] = {}
                
                # First, process any file links from the page (CSV, JSON, images, audio, etc.)
                from urllib.parse import urljoin
                
                # Extended list of processable file extensions
                PROCESSABLE_EXTENSIONS = [
                    '.csv', '.json', '.xlsx', '.txt', '.pdf', '.zip',  # Data files
                    '.png', '.jpg', '.jpeg', '.gif', '.webp',  # Images
                    '.opus', '.mp3', '.wav', '.ogg', '.m4a',   # Audio
                    '.mp4', '.webm',                            # Video
                ]
                
                if extracted.get("links"):
                    for link in extracted["links"]:
                        href = link.get("href", "")
                        if href and any(ext in href.lower() for ext in PROCESSABLE_EXTENSIONS):
                            abs_url = urljoin(current_url, href)
                            logger.info(f"   📁 Found file link: {href} → {abs_url}")
                            if abs_url not in question_components.data_sources:
                                question_components.data_sources.append(abs_url)
                
                # Add media sources (audio/video) to data sources for transcription
                if extracted.get("media_sources"):
                    for media_src in extracted["media_sources"]:
                        if media_src:
                            abs_url = urljoin(current_url, media_src)
                            logger.info(f"   🎵 Found media file: {media_src} → {abs_url}")
                            if abs_url not in question_components.data_sources:
                                question_components.data_sources.append(abs_url)
                
                # Store special values (like cutoff) in context
                if extracted.get("special_values"):
                    merged_context['special_values'] = extracted['special_values']
                    if 'cutoff' in extracted['special_values']:
                        merged_context['cutoff'] = extracted['special_values']['cutoff']
                        logger.info(f"   📊 Cutoff value: {merged_context['cutoff']}")
                
                # Scrape any relative URLs detected in question parsing
                if question_components.relative_urls:
                    for rel_url in question_components.relative_urls:
                        abs_url = urljoin(current_url, rel_url)
                        logger.info(f"   🌐 Scraping relative URL: {rel_url} → {abs_url}")
                        scraped_result = await scrape_url_with_browser(page, abs_url, session)
                        if scraped_result and scraped_result.get('text'):
                            merged_context['scraped_page'] = scraped_result['text']
                            merged_context['html_content'] = scraped_result.get('html', '')
                            logger.info(f"   ✓ Scraped {len(scraped_result['text'])} chars")
                
                # Check if we need to scrape additional URLs with the browser (from LLM)
                scrape_action = classification.get('scrape_action')
                if scrape_action and isinstance(scrape_action, dict):
                    scrape_url = scrape_action.get('url', '')
                    if scrape_url:
                        # Handle relative URLs
                        if scrape_url.startswith('/'):
                            scrape_url = urljoin(current_url, scrape_url)
                        
                        logger.info(f"   🌐 LLM requested browser scrape: {scrape_url}")
                        scraped_result = await scrape_url_with_browser(page, scrape_url, session)
                        if scraped_result and scraped_result.get('text'):
                            merged_context['scraped_page'] = scraped_result['text']
                            merged_context['html_content'] = scraped_result.get('html', '')
                            logger.info(f"   ✓ Browser scraped {len(scraped_result['text'])} chars")
                
                if question_components.data_sources:
                    logger.info(f"   Fetching {len(question_components.data_sources)} data source(s)...")
                    fetched_data = await fetch_all_data_sources(
                        question_components.data_sources, session
                    )
                    session.raw_data = fetched_data
                    
                    for source_url, data in fetched_data.items():
                        logger.info(f"   • {source_url[:60]}...")
                        if isinstance(data, dict):
                            if data.get('type') == 'audio':
                                merged_context['audio_transcript'] = data.get('transcript', '')
                                logger.info(f"     → Audio transcript: {len(data.get('transcript', ''))} chars")
                            elif data.get('type') == 'image':
                                # Store image data with dominant color
                                merged_context['image_description'] = data.get('description', '')
                                merged_context['dominant_color'] = data.get('dominant_color', '')
                                logger.info(f"     → Image: {len(data.get('description', ''))} chars, color: {data.get('dominant_color', 'N/A')}")
                            elif 'all_text' in data:
                                merged_context['pdf_text'] = data.get('all_text', '')
                                merged_context['pdf_tables'] = data.get('all_tables', [])
                                logger.info(f"     → PDF: {len(data.get('all_text', ''))} chars, {len(data.get('all_tables', []))} tables")
                            elif 'text' in data:
                                merged_context['webpage_text'] = data.get('text', '')
                                merged_context['webpage_tables'] = data.get('tables', [])
                                # Also capture raw HTML for CSS selector scraping
                                if data.get('html'):
                                    merged_context['html_content'] = data.get('html', '')
                                logger.info(f"     → Webpage: {len(data.get('text', ''))} chars, {len(data.get('tables', []))} tables")
                            elif 'owner' in data and 'repo' in data and 'sha' in data:
                                # GitHub tree config - store for API call
                                merged_context['github_config'] = data
                                logger.info(f"     → GitHub config: {data.get('owner')}/{data.get('repo')}")
                            elif data.get('type') == 'zip':
                                # ZIP file with logs data
                                merged_context['logs_data'] = data.get('logs_data', [])
                                merged_context['zip_files'] = data.get('files', {})
                                logger.info(f"     → ZIP: {len(data.get('logs_data', []))} log entries")
                            else:
                                merged_context[f'data_{len(merged_context)}'] = data
                                logger.info(f"     → Dict with keys: {list(data.keys())[:5]}")
                        elif isinstance(data, pd.DataFrame):
                            merged_context['dataframe'] = data
                            logger.info(f"     → DataFrame: {data.shape}")
                        elif isinstance(data, str):
                            merged_context['text_content'] = data
                            logger.info(f"     → Text: {len(data)} chars")
                        else:
                            merged_context[f'raw_{len(merged_context)}'] = data
                            logger.info(f"     → {type(data).__name__}")
                else:
                    logger.info("   No external data sources to fetch")
                
                # ==================== STAGE 6: DATA PREPARATION ====================
                logger.info("")
                logger.info("🧹 STAGE 6: Data Preparation")
                df: Optional[pd.DataFrame] = None
                
                # Get raw DataFrame without any transformations - let LLM handle it
                if 'dataframe' in merged_context:
                    df = merged_context['dataframe']
                    logger.info(f"   ✓ Raw DataFrame: {df.shape if df is not None else 'None'}")
                elif merged_context.get('pdf_tables'):
                    first_table = merged_context['pdf_tables'][0]
                    if first_table:
                        df = pd.DataFrame(first_table[1:], columns=first_table[0] if first_table else None)
                        logger.info(f"   ✓ DataFrame from PDF table: {df.shape if df is not None else 'None'}")
                elif merged_context.get('webpage_tables'):
                    if merged_context['webpage_tables']:
                        df = merged_context['webpage_tables'][0] if isinstance(merged_context['webpage_tables'][0], pd.DataFrame) else None
                        logger.info(f"   ✓ DataFrame from webpage table: {df.shape if df is not None else 'None'}")
                elif fetched_data:
                    # Get first fetched data that is a DataFrame
                    for source_url, data in fetched_data.items():
                        if isinstance(data, pd.DataFrame):
                            df = data
                            logger.info(f"   ✓ Raw DataFrame from {source_url}: {df.shape}")
                            break
                
                if df is None:
                    logger.info("   No structured DataFrame available")
                
                session.cleaned_data = df
                
                # ==================== STAGE 7: LLM-DRIVEN ANALYSIS ====================
                logger.info("")
                logger.info("🔬 STAGE 7: LLM-Driven Analysis & Answer Generation")
                logger.info("   (No hardcoded patterns - LLM decides everything)")
                
                # Import the LLM analysis module
                from .llm_analysis import solve_with_llm
                
                # Use LLM to analyze question and generate answer
                try:
                    answer = await solve_with_llm(
                        llm_client=llm_client,
                        question=raw_question,
                        context=merged_context,
                        df=df,
                        session=session
                    )
                    session.analysis_result = answer
                    logger.info(f"   ✓ LLM-driven answer: {str(answer)[:100]}...")
                except Exception as e:
                    logger.error(f"   ❌ LLM analysis failed: {e}")
                    # Ultimate fallback
                    answer = "start"
                    logger.info(f"   ✓ Using 'start' as fallback")
                
                # ==================== STAGE 8: FORMAT ANSWER ====================
                logger.info("")
                logger.info("📝 STAGE 8: Answer Formatting")
                formatted_answer = format_answer(
                    answer,
                    session.expected_format or AnswerFormat.STRING
                )
                session.final_answer = formatted_answer
                logger.info(f"   Answer: {formatted_answer}")
                logger.info(f"   ⏱️  Time used for this quiz: {time.time() - session.start_time:.1f}s")
                
                # Check for force submit
                if should_force_submit(session):
                    logger.warning(f"   ⏰ TIME CRITICAL: Only {time_remaining_safe(session):.0f}s left for this quiz!")
                
                # ==================== STAGE 9: SUBMIT ====================
                logger.info("")
                logger.info("📤 STAGE 9: Submission")
                
                # Resolve submit URL (handle relative URLs)
                from urllib.parse import urljoin, urlparse
                
                final_submit_url = session.submit_url  # Absolute URL from question if any
                
                if not final_submit_url and question_components.relative_submit_url:
                    # Resolve relative URL against base domain
                    parsed = urlparse(current_url)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    final_submit_url = f"{base_url}{question_components.relative_submit_url}"
                    logger.info(f"   Resolved relative submit URL: {question_components.relative_submit_url} → {final_submit_url}")
                
                submission_result = await submit_answer(
                    email=email,
                    secret=secret,
                    quiz_url=current_url,
                    answer=formatted_answer,
                    submit_url=final_submit_url,
                    session=session
                )
                
                session.submission_attempts.append({
                    "quiz_url": current_url,
                    "answer": str(formatted_answer),
                    "result": submission_result.model_dump()
                })
                
                logger.info(f"   Submit URL: {session.submit_url or 'default'}")
                logger.info(f"   Response: correct={submission_result.correct}, reason={submission_result.reason or 'none'}")
                
                # ==================== STAGE 10: PROCESS RESULT ====================
                logger.info("")
                if submission_result.correct:
                    logger.info("✅ CORRECT!")
                    if submission_result.url:
                        # Resolve relative URLs to absolute
                        next_url = submission_result.url
                        if not next_url.startswith('http'):
                            next_url = urljoin(current_url, next_url)
                        logger.info(f"   → Next quiz: {next_url}")
                        current_url = next_url
                        session.quiz_chain.append(current_url)
                        session.current_url = current_url
                    else:
                        logger.info("🎉 QUIZ CHAIN COMPLETE!")
                        break
                else:
                    logger.warning(f"❌ INCORRECT: {submission_result.reason}")
                    
                    if submission_result.url:
                        # Resolve relative URLs to absolute
                        next_url = submission_result.url
                        if not next_url.startswith('http'):
                            next_url = urljoin(current_url, next_url)
                        logger.info(f"   → Moving to: {next_url}")
                        current_url = next_url
                        session.quiz_chain.append(current_url)
                    else:
                        logger.info("   No more quizzes to attempt")
                        break
                
            except Exception as e:
                logger.error(f"❌ Quiz {quiz_count} failed: {e}", exc_info=True)
                session.error_log.append(str(e))
                break
        
        # ==================== FINAL SUMMARY ====================
        logger.info("")
        logger.info("=" * 70)
        logger.info("🏁 PIPELINE COMPLETE")
        logger.info(f"   Quizzes attempted: {quiz_count}")
        logger.info(f"   Quiz chain: {' → '.join(session.quiz_chain[:5])}")
        logger.info(f"   Total time: {time.time() - session.start_time:.1f}s")
        logger.info(f"   Submissions: {len(session.submission_attempts)}")
        for i, sub in enumerate(session.submission_attempts, 1):
            result = sub.get('result', {})
            status = '✅' if result.get('correct') else '❌'
            logger.info(f"     {i}. {status} Answer: {sub.get('answer', 'N/A')[:50]}")
        logger.info("=" * 70)
        
        log_step(session, "pipeline_complete", {
            "quizzes_solved": quiz_count,
            "total_time": time.time() - session.start_time
        })
    
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        log_step(session, "pipeline_error", {"error": str(e)})
    
    finally:
        logger.info("🔒 Closing browser...")
        await browser_manager.close()
        logger.info("   ✓ Browser closed")


async def run_quiz_background(email: str, secret: str, url: str) -> None:
    """Background task wrapper for quiz solving."""
    
    deadline = time.time() + settings.timeouts.quiz_deadline_seconds
    
    try:
        await solve_quiz_pipeline(email, secret, url, deadline)
    except Exception as e:
        logger.error(f"Background quiz task failed: {e}", exc_info=True)
