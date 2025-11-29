#!/usr/bin/env python3
"""
Integration tests with REAL web scraping.
These tests actually fetch data from the internet to verify scraping works.

Run: uv run python tests/test_real_scraping.py
"""

import asyncio
import time
from typing import Any

# Test results tracking
results: list[dict[str, Any]] = []


def log_result(test_name: str, passed: bool, details: str = ""):
    """Log a test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"       {details}")
    results.append({"test": test_name, "passed": passed, "details": details})


async def test_wikipedia_static_scraping():
    """Test static scraping on a simple public HTML page (Wikipedia blocks bots)"""
    print("\n" + "=" * 60)
    print("TEST 1: Static HTML Scraping (W3Schools HTML Tables)")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import scrape_webpage_static
    
    # Use a more permissive site that allows scraping
    # W3Schools example page with tables
    url = "https://www.w3schools.com/html/html_tables.asp"
    
    try:
        start = time.time()
        result = await scrape_webpage_static(url)
        elapsed = time.time() - start
        
        print(f"  Scrape time: {elapsed:.2f}s")
        print(f"  Text length: {len(result.get('text', ''))} chars")
        print(f"  Tables found: {len(result.get('tables', []))}")
        print(f"  Lists found: {len(result.get('lists', []))}")
        
        # Verify we got content
        has_text = len(result.get('text', '')) > 100
        has_tables_or_lists = len(result.get('tables', [])) > 0 or len(result.get('lists', [])) > 0
        
        if has_text:
            text_preview = result['text'][:200].replace('\n', ' ')
            print(f"  Text preview: {text_preview}...")
        
        assert has_text, "Text too short"
        
        log_result("Static HTML scraping (W3Schools)", True, f"{len(result.get('text', ''))} chars in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        log_result("Static HTML scraping (W3Schools)", False, str(e))
        return False


async def test_json_api_fetch():
    """Test fetching JSON from a public API"""
    print("\n" + "=" * 60)
    print("TEST 2: JSON API Fetch (JSONPlaceholder)")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import call_api
    
    url = "https://jsonplaceholder.typicode.com/users"
    
    try:
        start = time.time()
        result = await call_api(url)
        elapsed = time.time() - start
        
        print(f"  Fetch time: {elapsed:.2f}s")
        print(f"  Result type: {type(result)}")
        
        if isinstance(result, list):
            print(f"  Items count: {len(result)}")
            if result:
                print(f"  First item keys: {list(result[0].keys())}")
                print(f"  First user: {result[0].get('name')}")
        
        assert isinstance(result, list), "Expected list response"
        assert len(result) == 10, f"Expected 10 users, got {len(result)}"
        assert result[0].get('name'), "No name field in user"
        
        log_result("JSON API fetch", True, f"{len(result)} items in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        log_result("JSON API fetch", False, str(e))
        return False


async def test_csv_download():
    """Test downloading and parsing a CSV file"""
    print("\n" + "=" * 60)
    print("TEST 3: CSV File Download & Parse")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import download_and_parse_file
    
    # Public CSV dataset
    url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
    
    try:
        start = time.time()
        df = await download_and_parse_file(url)
        elapsed = time.time() - start
        
        print(f"  Download time: {elapsed:.2f}s")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample data:\n{df.head(3)}")
        
        assert df is not None, "No DataFrame returned"
        assert len(df) > 100, "Too few rows"
        assert 'Country' in df.columns or 'country' in df.columns.str.lower().tolist(), "No Country column"
        
        log_result("CSV download & parse", True, f"Shape {df.shape} in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        log_result("CSV download & parse", False, str(e))
        return False


async def test_github_raw_json():
    """Test fetching JSON from GitHub raw content"""
    print("\n" + "=" * 60)
    print("TEST 4: GitHub Raw JSON Fetch")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import download_and_parse_file
    
    # A known public JSON file
    url = "https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv"
    
    try:
        start = time.time()
        df = await download_and_parse_file(url)
        elapsed = time.time() - start
        
        print(f"  Download time: {elapsed:.2f}s")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns (first 5): {list(df.columns)[:5]}")
        
        assert df is not None, "No data returned"
        assert len(df) > 100, "Too few countries"
        
        log_result("GitHub raw file fetch", True, f"Shape {df.shape} in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        log_result("GitHub raw file fetch", False, str(e))
        return False


async def test_httpbin_api():
    """Test API with headers using httpbin.org"""
    print("\n" + "=" * 60)
    print("TEST 5: HTTPBin API Test")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import call_api
    
    url = "https://httpbin.org/get"
    
    try:
        start = time.time()
        result = await call_api(url)
        elapsed = time.time() - start
        
        print(f"  Fetch time: {elapsed:.2f}s")
        print(f"  Response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
        if isinstance(result, dict):
            print(f"  Origin IP: {result.get('origin', 'N/A')}")
            print(f"  User-Agent: {result.get('headers', {}).get('User-Agent', 'N/A')[:50]}...")
        
        assert isinstance(result, dict), "Expected dict response"
        assert 'origin' in result, "No origin in response"
        
        log_result("HTTPBin API test", True, f"Response in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        log_result("HTTPBin API test", False, str(e))
        return False


async def test_simple_html_scrape():
    """Test scraping a simple HTML page"""
    print("\n" + "=" * 60)
    print("TEST 6: Simple HTML Scrape (Example.com)")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import scrape_webpage
    
    url = "https://example.com"
    
    try:
        start = time.time()
        html = await scrape_webpage(url)
        elapsed = time.time() - start
        
        print(f"  Scrape time: {elapsed:.2f}s")
        print(f"  HTML length: {len(html)} chars")
        print(f"  Contains 'Example Domain': {'Example Domain' in html}")
        
        assert len(html) > 100, "HTML too short"
        assert "Example Domain" in html, "Expected content not found"
        
        log_result("Simple HTML scrape", True, f"{len(html)} chars in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        log_result("Simple HTML scrape", False, str(e))
        return False


async def test_user_agent_rotation():
    """Test that user agent rotation is working"""
    print("\n" + "=" * 60)
    print("TEST 7: User-Agent Rotation")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import get_random_user_agent, USER_AGENTS
    
    try:
        agents = set()
        for _ in range(20):
            ua = get_random_user_agent()
            agents.add(ua)
        
        print(f"  Total UA pool: {len(USER_AGENTS)}")
        print(f"  Unique UAs in 20 calls: {len(agents)}")
        
        assert len(USER_AGENTS) >= 4, "Need at least 4 user agents"
        assert len(agents) >= 2, "User agent should vary"
        
        log_result("User-Agent rotation", True, f"{len(agents)} unique agents used")
        return True
        
    except Exception as e:
        log_result("User-Agent rotation", False, str(e))
        return False


async def test_table_extraction_from_html():
    """Test extracting tables from HTML content"""
    print("\n" + "=" * 60)
    print("TEST 8: HTML Table Extraction")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import extract_html_tables
    from bs4 import BeautifulSoup
    
    # Sample HTML with a table
    html = """
    <html>
    <body>
        <table>
            <tr><th>Name</th><th>Value</th><th>Count</th></tr>
            <tr><td>Item A</td><td>100</td><td>5</td></tr>
            <tr><td>Item B</td><td>200</td><td>10</td></tr>
            <tr><td>Item C</td><td>300</td><td>15</td></tr>
        </table>
    </body>
    </html>
    """
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        tables = extract_html_tables(soup)
        
        print(f"  Tables extracted: {len(tables)}")
        
        if tables:
            df = tables[0]
            print(f"  Table shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sum of 'Value' column: {df['Value'].astype(int).sum() if 'Value' in df.columns else 'N/A'}")
        
        assert len(tables) == 1, "Should find 1 table"
        assert tables[0].shape == (3, 3), f"Wrong shape: {tables[0].shape}"
        
        log_result("HTML table extraction", True, f"Extracted {len(tables)} table(s)")
        return True
        
    except Exception as e:
        log_result("HTML table extraction", False, str(e))
        return False


async def test_data_cleaning():
    """Test data cleaning and preparation"""
    print("\n" + "=" * 60)
    print("TEST 9: Data Cleaning")
    print("=" * 60)
    
    from quiz_solver.data_sourcing import clean_and_prepare_data
    import pandas as pd
    
    try:
        # Test with messy data
        raw_data = [
            {"Name": "  Item A  ", "VALUE": "100", "Count": 5},
            {"Name": "Item B", "VALUE": "200", "Count": None},
            {"Name": "Item C", "VALUE": "300", "Count": 15},
            {"Name": "Item C", "VALUE": "300", "Count": 15},  # Duplicate
        ]
        
        df = clean_and_prepare_data(raw_data, "test question")
        
        print(f"  Original rows: 4, Cleaned rows: {len(df)}")
        print(f"  Columns (normalized): {list(df.columns)}")
        print(f"  Missing values filled: {df.isnull().sum().sum() == 0}")
        
        assert len(df) == 3, "Duplicates should be removed"
        assert all(c.islower() for c in df.columns), "Columns should be lowercase"
        assert df.isnull().sum().sum() == 0, "No null values should remain"
        
        log_result("Data cleaning", True, "Duplicates removed, nulls filled, columns normalized")
        return True
        
    except Exception as e:
        log_result("Data cleaning", False, str(e))
        return False


async def test_analysis_sum():
    """Test sum analysis on data"""
    print("\n" + "=" * 60)
    print("TEST 10: Analysis - Sum Calculation")
    print("=" * 60)
    
    from quiz_solver.analysis import simple_analysis
    import pandas as pd
    
    try:
        df = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [100, 200, 300, 400, 500],
            'quantity': [1, 2, 3, 4, 5]
        })
        
        # Test sum
        result = simple_analysis(df, "What is the sum of the value column?")
        print(f"  Sum of 'value': {result} (expected: 1500)")
        assert result == 1500, f"Expected 1500, got {result}"
        
        # Test count
        result = simple_analysis(df, "How many items are there?")
        print(f"  Count: {result} (expected: 5)")
        assert result == 5, f"Expected 5, got {result}"
        
        # Test average
        result = simple_analysis(df, "What is the average value?")
        print(f"  Average: {result} (expected: 300)")
        assert result == 300, f"Expected 300, got {result}"
        
        log_result("Analysis calculations", True, "Sum, count, average all correct")
        return True
        
    except Exception as e:
        log_result("Analysis calculations", False, str(e))
        return False


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("🧪 REAL-WORLD SCRAPING INTEGRATION TESTS")
    print("=" * 60)
    print("These tests make actual HTTP requests to verify scraping works.\n")
    
    start_time = time.time()
    
    # Run all tests
    await test_wikipedia_static_scraping()
    await test_json_api_fetch()
    await test_csv_download()
    await test_github_raw_json()
    await test_httpbin_api()
    await test_simple_html_scrape()
    await test_user_agent_rotation()
    await test_table_extraction_from_html()
    await test_data_cleaning()
    await test_analysis_sum()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed
    
    for r in results:
        status = "✅" if r['passed'] else "❌"
        print(f"  {status} {r['test']}")
    
    print(f"\nTotal: {passed}/{len(results)} passed, {failed} failed")
    print(f"Time: {elapsed:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed - check logs above")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
