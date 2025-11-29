"""
Direct test for Quiz 3 (demo-audio) to debug CSV and cutoff handling.
"""

import asyncio
import aiohttp
import pandas as pd
import io
from urllib.parse import urljoin

# Test Quiz 3 URL
QUIZ_URL = "https://tds-llm-analysis.s-anand.net/demo-audio?email=24f100161@ds.study.iitm.ac.in&id=33536"


async def test_quiz3():
    """Test Quiz 3 directly"""
    
    print("=" * 60)
    print("Testing Quiz 3: demo-audio")
    print("=" * 60)
    
    # Step 1: Fetch the quiz page
    print("\n1. Fetching quiz page...")
    async with aiohttp.ClientSession() as session:
        async with session.get(QUIZ_URL) as resp:
            html = await resp.text()
            print(f"   Page fetched: {len(html)} chars")
            print(f"   Preview: {html[:500]}...")
    
    # Step 2: Parse for cutoff value and CSV link
    print("\n2. Parsing page for cutoff and CSV...")
    import re
    
    # Look for cutoff value
    cutoff_match = re.search(r'id=["\']cutoff["\'][^>]*>(\d+)', html)
    if cutoff_match:
        cutoff = int(cutoff_match.group(1))
        print(f"   Found cutoff: {cutoff}")
    else:
        # Try another pattern
        cutoff_match = re.search(r'>(\d+)</span>\s*</p>', html)
        if cutoff_match:
            cutoff = int(cutoff_match.group(1))
            print(f"   Found cutoff (alt pattern): {cutoff}")
        else:
            print("   WARNING: Could not find cutoff value!")
            cutoff = None
    
    # Look for CSV link
    csv_match = re.search(r'href="([^"]+\.csv)"', html)
    if csv_match:
        csv_relative = csv_match.group(1)
        csv_url = urljoin(QUIZ_URL, csv_relative)
        print(f"   Found CSV link: {csv_relative} -> {csv_url}")
    else:
        print("   WARNING: Could not find CSV link!")
        csv_url = None
    
    # Step 3: Download and parse CSV
    if csv_url:
        print("\n3. Downloading CSV...")
        async with aiohttp.ClientSession() as session:
            async with session.get(csv_url) as resp:
                csv_content = await resp.read()
                print(f"   CSV downloaded: {len(csv_content)} bytes")
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(csv_content))
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   First 5 rows:\n{df.head()}")
        print(f"   Data types: {df.dtypes.to_dict()}")
    else:
        print("\n3. Skipping CSV download (no URL found)")
        df = None
    
    # Step 4: Calculate sum of values > cutoff
    if df is not None and cutoff is not None:
        print(f"\n4. Computing sum of values > {cutoff}...")
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"   Numeric columns: {numeric_cols}")
        
        for col in numeric_cols:
            values = df[col]
            filtered = values[values > cutoff]
            total = filtered.sum()
            print(f"   Column '{col}': {len(filtered)} values > {cutoff}, sum = {total}")
        
        # Assume answer is sum of first numeric column
        if numeric_cols:
            answer_col = numeric_cols[0]
            answer = int(df[df[answer_col] > cutoff][answer_col].sum())
            print(f"\n   ANSWER: {answer}")
    else:
        print("\n4. Cannot compute sum (missing data)")


if __name__ == "__main__":
    asyncio.run(test_quiz3())
