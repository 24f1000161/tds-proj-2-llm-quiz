#!/usr/bin/env python3
"""
Test the quiz solver with the actual demo endpoint.
"""

import asyncio
import httpx


async def test_demo_quiz():
    """Send a test request to the demo endpoint."""
    
    payload = {
        "email": "24f100161@ds.study.iitm.ac.in",
        "secret": "iamdivyam",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
    
    print("=" * 60)
    print("🧪 Testing Quiz Solver with Demo Endpoint")
    print("=" * 60)
    print(f"Email: {payload['email']}")
    print(f"URL: {payload['url']}")
    print()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("Sending POST request to http://localhost:8000/api/quiz...")
            response = await client.post(
                "http://localhost:8000/api/quiz",
                json=payload
            )
            
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Body: {response.json()}")
            
            if response.status_code == 200:
                print("\n✅ Request accepted! Quiz solver is processing in background.")
                print("Check the server logs for quiz solving progress.")
            else:
                print(f"\n❌ Request failed with status {response.status_code}")
                
    except httpx.ConnectError:
        print("\n❌ Could not connect to server. Make sure it's running on port 8000.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_demo_quiz())
