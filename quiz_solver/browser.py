"""
Browser automation using Playwright for handling Vue.js/React/Angular SPAs.
"""

import base64
from typing import Optional, Any
from html import unescape
from urllib.parse import unquote

from playwright.async_api import async_playwright, Browser, Page, BrowserContext, Playwright

from .config import settings
from .logging_utils import logger, log_step


class BrowserManager:
    """Manages Playwright browser for SPA navigation."""
    
    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
    
    async def launch(self) -> Page:
        """Launch browser and return page."""
        
        self._playwright = await async_playwright().start()
        
        self._browser = await self._playwright.chromium.launch(
            headless=settings.browser.headless,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
        
        self._context = await self._browser.new_context(
            user_agent=settings.browser.user_agent,
            ignore_https_errors=True,
            viewport={
                "width": settings.browser.viewport_width,
                "height": settings.browser.viewport_height
            }
        )
        
        self._page = await self._context.new_page()
        
        # Set timeouts
        self._page.set_default_timeout(settings.timeouts.navigation_timeout)
        self._page.set_default_navigation_timeout(settings.timeouts.navigation_timeout)
        
        logger.info("Browser launched successfully")
        return self._page
    
    async def close(self) -> None:
        """Close browser and cleanup."""
        
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        logger.info("Browser closed")
    
    @property
    def page(self) -> Optional[Page]:
        """Get current page."""
        return self._page


async def navigate_to_quiz(page: Page, url: str, session: Any) -> None:
    """Navigate to quiz URL and wait for full rendering."""
    
    try:
        response = await page.goto(
            url,
            wait_until="networkidle",
            timeout=settings.timeouts.navigation_timeout
        )
        
        if response and response.status != 200:
            logger.warning(f"Got status {response.status} for {url}")
        
        log_step(session, "navigation_complete", {
            "url": url,
            "status": response.status if response else "unknown"
        })
        
    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        raise


async def wait_for_spa_rendering(page: Page, session: Any) -> None:
    """Wait for modern framework (Vue/React/Angular) to render."""
    
    FRAMEWORK_CHECKS = {
        "vue": {
            "wait_selector": "[data-v-app], [id='app'], #app",
            "detection": "typeof window !== 'undefined' && (window.__VUE__ !== undefined || window.__VUE_DEVTOOLS_GLOBAL_HOOK__ !== undefined)"
        },
        "react": {
            "wait_selector": "[data-reactroot], #root",
            "detection": "typeof window !== 'undefined' && window.__REACT_DEVTOOLS_GLOBAL_HOOK__ !== undefined"
        },
        "angular": {
            "wait_selector": "[ng-app], app-root",
            "detection": "typeof window !== 'undefined' && window.ng !== undefined"
        }
    }
    
    detected_framework = None
    
    # Strategy 1: Check for framework-specific DOM elements
    for framework, config in FRAMEWORK_CHECKS.items():
        try:
            await page.wait_for_selector(config["wait_selector"], timeout=1000)
            detected_framework = framework
            logger.info(f"Detected {framework} framework")
            break
        except Exception:
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
            except Exception:
                pass
    
    # Strategy 3: Wait for content in #result div (short timeout - most quizzes don't have this)
    try:
        await page.wait_for_function(
            """() => {
                const element = document.querySelector("#result");
                return element && element.textContent.trim().length > 0;
            }""",
            timeout=1000  # Reduced from 5000ms to 1000ms - most quizzes don't need this
        )
        logger.info("Content detected in #result div")
    except Exception:
        # This is expected for most quizzes - continue normally
        pass
    
    # Skip waiting for networkidle - it's too slow
    # await page.wait_for_load_state("networkidle")
    
    log_step(session, "spa_rendered", {
        "framework": detected_framework or "unknown"
    })


async def extract_and_decode_content(page: Page, session: Any) -> dict[str, Any]:
    """Extract all content from page and handle base64 decoding - fast version."""
    
    # Set a short timeout for all operations
    page.set_default_timeout(5000)  # 5 seconds max
    
    # Get full page HTML
    html_content = await page.content()
    
    # Get final rendered text
    final_text = await page.evaluate("document.body.innerText")
    
    # Get the #result div specifically
    result_content = None
    try:
        result_el = page.locator("#result")
        if await result_el.count() > 0:
            result_content = await result_el.first.inner_html(timeout=2000)
    except Exception:
        pass
    
    # Extract all links (href) from anchor tags
    links = []
    try:
        links = await page.evaluate("""() => {
            const anchors = document.querySelectorAll('a[href]');
            return Array.from(anchors).map(a => ({
                href: a.getAttribute('href'),
                text: a.innerText.trim()
            }));
        }""")
    except Exception:
        pass
    
    # Extract audio/video sources
    media_sources = []
    try:
        media_sources = await page.evaluate("""() => {
            const media = document.querySelectorAll('audio[src], video[src], source[src]');
            return Array.from(media).map(m => m.getAttribute('src'));
        }""")
    except Exception:
        pass
    
    # Extract specific values like cutoff from spans - with short timeout
    special_values = {}
    try:
        cutoff_el = page.locator("#cutoff")
        if await cutoff_el.count() > 0:
            cutoff_text = await cutoff_el.first.inner_text(timeout=2000)
            if cutoff_text:
                special_values['cutoff'] = cutoff_text
    except Exception:
        pass
    
    extracted = {
        "full_html": html_content,
        "text_content": final_text,
        "result_div_html": result_content,
        "page_url": page.url,
        "links": links,
        "media_sources": media_sources,
        "special_values": special_values
    }
    
    log_step(session, "content_extracted", {
        "html_length": len(html_content),
        "text_length": len(final_text),
        "has_result_div": result_content is not None,
        "links_count": len(links),
        "media_count": len(media_sources),
        "special_values": special_values
    })
    
    return extracted


def decode_content_multi_layer(content: str) -> tuple[str, int]:
    """Handle multiple encoding layers (base64, URL, HTML entities)."""
    
    decoded_content = content
    layers_decoded = 0
    
    # Try up to 5 layers of base64 encoding
    for attempt in range(5):
        try:
            # Skip if already looks like normal text
            if decoded_content.isprintable() and '\n' in decoded_content and len(decoded_content) > 100:
                break
            
            decoded = base64.b64decode(decoded_content).decode('utf-8')
            if decoded != decoded_content and len(decoded) > 0:
                decoded_content = decoded
                layers_decoded += 1
                logger.debug(f"Base64 decode layer {layers_decoded} successful")
            else:
                break
        except Exception:
            break
    
    # URL decode
    try:
        url_decoded = unquote(decoded_content)
        if url_decoded != decoded_content:
            decoded_content = url_decoded
            logger.debug("URL decode successful")
    except Exception:
        pass
    
    # Handle HTML entities
    try:
        entity_decoded = unescape(decoded_content)
        if entity_decoded != decoded_content:
            decoded_content = entity_decoded
            logger.debug("HTML entity decode successful")
    except Exception:
        pass
    
    return decoded_content, layers_decoded


async def handle_spa_edge_cases(page: Page, session: Any) -> None:
    """Handle common SPA edge cases - fast version."""
    
    # Edge Case 1: Loading spinners/overlays - quick check only
    try:
        spinners = await page.locator(".loading, .spinner, [data-loading='true']").all()
        for spinner in spinners[:2]:  # Only check first 2
            try:
                await spinner.wait_for(state="hidden", timeout=2000)
            except Exception:
                pass
    except Exception:
        pass
    
    # Scroll to top
    try:
        await page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass


async def handle_pagination(page: Page, session: Any, max_pages: int = 10) -> list[str]:
    """
    Handle pagination by clicking Next buttons.
    Returns list of all page contents collected.
    """
    import asyncio
    import random
    
    all_content: list[str] = []
    pages_scraped = 1
    
    # Collect initial page content
    try:
        initial_content = await page.evaluate("document.body.innerText")
        all_content.append(initial_content)
    except Exception:
        pass
    
    # Pagination button selectors (in priority order)
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
        "button.next",
        "a:has-text('>')",
        "button:has-text('>')",
        "a:has-text('»')",
    ]
    
    while pages_scraped < max_pages:
        clicked = False
        
        for selector in next_selectors:
            try:
                next_btn = page.locator(selector).first
                if await next_btn.count() > 0 and await next_btn.is_visible():
                    # Check if button is enabled
                    is_disabled = await next_btn.get_attribute("disabled")
                    aria_disabled = await next_btn.get_attribute("aria-disabled")
                    
                    if is_disabled or aria_disabled == "true":
                        continue
                    
                    # Random delay for anti-blocking
                    await asyncio.sleep(random.uniform(0.3, 1.0))
                    
                    await next_btn.click()
                    await page.wait_for_load_state("networkidle")
                    
                    # Collect new page content
                    try:
                        page_content = await page.evaluate("document.body.innerText")
                        all_content.append(page_content)
                    except Exception:
                        pass
                    
                    pages_scraped += 1
                    clicked = True
                    logger.info(f"Pagination: scraped page {pages_scraped}")
                    break
            except Exception:
                continue
        
        if not clicked:
            # No more pages or pagination not found
            break
    
    if pages_scraped > 1:
        log_step(session, "pagination_complete", {"pages_scraped": pages_scraped})
    
    return all_content


async def click_show_more(page: Page) -> int:
    """
    Click 'Show More' / 'Load More' buttons to reveal hidden content.
    Returns number of clicks performed.
    """
    import asyncio
    import random
    
    reveal_selectors = [
        "button:has-text('Show More')",
        "button:has-text('View More')",
        "button:has-text('Load More')",
        "button:has-text('See More')",
        "button:has-text('Expand')",
        "a:has-text('Show More')",
        "a:has-text('Load More')",
        "[data-toggle='collapse']",
        ".show-more",
        ".view-more",
        ".expand-btn",
        ".load-more",
    ]
    
    clicks = 0
    max_clicks = 5
    
    for selector in reveal_selectors:
        if clicks >= max_clicks:
            break
        
        try:
            elements = await page.locator(selector).all()
            for element in elements[:3]:  # Limit to 3 per selector
                if clicks >= max_clicks:
                    break
                try:
                    if await element.is_visible():
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        await element.click()
                        clicks += 1
                        await asyncio.sleep(0.5)  # Wait for content to load
                except Exception:
                    pass
        except Exception:
            pass
    
    if clicks > 0:
        logger.info(f"Clicked {clicks} 'Show More' buttons")
    
    return clicks
