"""
Test Quiz 3 with browser rendering to get dynamic cutoff value.
"""

import asyncio
from playwright.async_api import async_playwright


QUIZ_URL = "https://tds-llm-analysis.s-anand.net/demo-audio?email=24f100161@ds.study.iitm.ac.in&id=33536"


async def test_with_browser():
    """Test with Playwright to get dynamic content"""
    
    print("=" * 60)
    print("Testing Quiz 3 with Playwright (for dynamic cutoff)")
    print("=" * 60)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("\n1. Navigating to quiz page...")
        await page.goto(QUIZ_URL, wait_until="networkidle")
        
        # Wait for JavaScript to execute
        await asyncio.sleep(2)
        
        print("\n2. Extracting rendered content...")
        
        # Get cutoff from rendered page
        try:
            cutoff_el = page.locator("#cutoff")
            cutoff_text = await cutoff_el.inner_text()
            print(f"   Cutoff (from #cutoff): '{cutoff_text}'")
        except Exception as e:
            print(f"   Error getting cutoff: {e}")
            cutoff_text = None
        
        # Get all page text
        page_text = await page.evaluate("document.body.innerText")
        print(f"\n3. Full page text:\n{page_text[:1000]}")
        
        # Get links
        links = await page.evaluate("""() => {
            const anchors = document.querySelectorAll('a[href]');
            return Array.from(anchors).map(a => ({
                href: a.getAttribute('href'),
                text: a.innerText.trim()
            }));
        }""")
        print(f"\n4. Links: {links}")
        
        await browser.close()
        
        return cutoff_text


if __name__ == "__main__":
    asyncio.run(test_with_browser())
