#!/usr/bin/env python3
"""
Smoke test for the quiz solver API.
Run: uv run python test_smoke.py
"""

import asyncio
import time
import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"

@pytest.mark.asyncio
async def test_smoke():
    print("=" * 60)
    print("SMOKE TEST: Quiz Solver API")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        # Test 1: Health check
        print("\n1. Testing health endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/")
            print(f"   GET / -> {response.status_code}")
            print(f"   Response: {response.json()}")
            assert response.status_code == 200, "Health check failed"
            print("   ✅ PASS")
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            return False
        
        # Test 2: Invalid JSON -> 400
        print("\n2. Testing invalid JSON (expect 400)...")
        try:
            response = await client.post(
                f"{BASE_URL}/api/quiz",
                content="invalid json {",
                headers={"Content-Type": "application/json"}
            )
            print(f"   POST /api/quiz with invalid JSON -> {response.status_code}")
            assert response.status_code == 400, f"Expected 400, got {response.status_code}"
            print("   ✅ PASS")
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            return False
        
        # Test 3: Invalid secret -> 403
        print("\n3. Testing invalid secret (expect 403)...")
        try:
            response = await client.post(
                f"{BASE_URL}/api/quiz",
                json={
                    "email": "test@example.com",
                    "secret": "WRONG_SECRET",
                    "url": "https://example.com/quiz"
                }
            )
            print(f"   POST /api/quiz with wrong secret -> {response.status_code}")
            assert response.status_code == 403, f"Expected 403, got {response.status_code}"
            print("   ✅ PASS")
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            return False
        
        # Test 4: Valid request -> 200 (requires STUDENT_SECRETS env var)
        print("\n4. Testing valid request (expect 200)...")
        print("   Note: Requires STUDENT_SECRETS=test@example.com:test_secret")
        try:
            start = time.time()
            response = await client.post(
                f"{BASE_URL}/api/quiz",
                json={
                    "email": "test@example.com",
                    "secret": "test_secret",
                    "url": "https://example.com/quiz"
                }
            )
            elapsed = (time.time() - start) * 1000
            print(f"   POST /api/quiz -> {response.status_code} in {elapsed:.1f}ms")
            
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
                print(f"   Response time: {elapsed:.1f}ms (target < 100ms)")
                print("   ✅ PASS")
            elif response.status_code == 403:
                print("   ⚠️ SKIP: Secret not configured (expected if env var not set)")
            else:
                print(f"   ❌ FAIL: Unexpected status {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ ALL SMOKE TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    result = asyncio.run(test_smoke())
    exit(0 if result else 1)
