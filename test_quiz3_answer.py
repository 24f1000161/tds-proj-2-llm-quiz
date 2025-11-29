"""
Complete Quiz 3 test with answer computation.
"""

import asyncio
import aiohttp
import pandas as pd
import io
from urllib.parse import urljoin


async def compute_quiz3_answer():
    """Compute the actual answer for Quiz 3"""
    
    print("=" * 60)
    print("Computing Quiz 3 Answer")
    print("=" * 60)
    
    # Known values from browser test
    cutoff = 20409
    csv_url = "https://tds-llm-analysis.s-anand.net/demo-audio-data.csv"
    
    print(f"\n1. Cutoff: {cutoff}")
    print(f"2. CSV URL: {csv_url}")
    
    # Download CSV
    print("\n3. Downloading CSV...")
    async with aiohttp.ClientSession() as session:
        async with session.get(csv_url) as resp:
            csv_content = await resp.read()
            print(f"   Downloaded: {len(csv_content)} bytes")
    
    # Parse CSV
    df = pd.read_csv(io.BytesIO(csv_content))
    print(f"\n4. DataFrame info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   First column header: '{df.columns[0]}'")
    
    # Note: The first row is being used as header!
    # The column name '96903' is actually a data value
    # Let's read without header
    df_no_header = pd.read_csv(io.BytesIO(csv_content), header=None)
    print(f"\n5. DataFrame without header:")
    print(f"   Shape: {df_no_header.shape}")
    print(f"   First 5 rows:\n{df_no_header.head()}")
    
    # All values are in column 0
    all_values = df_no_header[0]
    print(f"\n6. All values (column 0):")
    print(f"   Count: {len(all_values)}")
    print(f"   Min: {all_values.min()}")
    print(f"   Max: {all_values.max()}")
    print(f"   Sum: {all_values.sum()}")
    
    # Filter values > cutoff
    filtered = all_values[all_values > cutoff]
    print(f"\n7. Values > {cutoff}:")
    print(f"   Count: {len(filtered)}")
    print(f"   Sum: {int(filtered.sum())}")
    
    answer = int(filtered.sum())
    print(f"\n🎯 ANSWER: {answer}")
    
    return answer


if __name__ == "__main__":
    asyncio.run(compute_quiz3_answer())
