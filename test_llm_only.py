"""Quick test to verify LLM API connections work."""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_gemini_via_aipipe():
    """Test Gemini via aipipe's native endpoint."""
    import httpx
    
    token = os.getenv("AIPIPE_TOKEN")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    
    print(f"\n{'='*60}")
    print(f"Testing Gemini via aipipe")
    print(f"  Token: {token[:20]}...{token[-10:]}")
    print(f"  Model: {model}")
    print(f"{'='*60}")
    
    url = f"https://aipipe.org/geminiv1beta/models/{model}:generateContent"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "What is 2 + 2? Reply with just the number."}
                ]
            }
        ]
    }
    
    headers = {
        "x-goog-api-key": token,
        "Content-Type": "application/json"
    }
    
    print(f"\nRequest URL: {url}")
    print(f"Request headers: x-goog-api-key: {token[:20]}...")
    print(f"Request payload: {payload}")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            print(f"\nResponse status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse data: {data}")
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                print(f"\n✅ SUCCESS! Response: {content}")
                return True
            else:
                print(f"\n❌ FAILED! Status: {response.status_code}")
                print(f"Response body: {response.text}")
                return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


async def test_openai_via_aipipe():
    """Test OpenAI models via aipipe's OpenRouter endpoint."""
    import httpx
    
    token = os.getenv("AIPIPE_TOKEN")
    model = os.getenv("AIPIPE_MODEL", "openai/gpt-4.1-nano")
    
    # Ensure model has provider prefix
    if "/" not in model:
        model = f"openai/{model}"
    
    print(f"\n{'='*60}")
    print(f"Testing OpenAI via aipipe (OpenRouter endpoint)")
    print(f"  Token: {token[:20]}...{token[-10:]}")
    print(f"  Model: {model}")
    print(f"{'='*60}")
    
    # Use OpenRouter endpoint, NOT /openai/v1
    url = "https://aipipe.org/openrouter/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print(f"\nRequest URL: {url}")
    print(f"Request payload: {payload}")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            print(f"\nResponse status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"\n✅ SUCCESS! Response: {content}")
                return True
            else:
                print(f"\n❌ FAILED! Status: {response.status_code}")
                print(f"Response body: {response.text}")
                return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


async def main():
    print("\n" + "="*60)
    print("LLM API Connection Tests")
    print("="*60)
    
    # Test Gemini via aipipe first (should work with correct endpoint)
    gemini_ok = await test_gemini_via_aipipe()
    
    # Test OpenAI via aipipe
    openai_ok = await test_openai_via_aipipe()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Gemini via aipipe: {'✅ OK' if gemini_ok else '❌ FAILED'}")
    print(f"  OpenAI via aipipe: {'✅ OK' if openai_ok else '❌ FAILED'}")
    
    if gemini_ok or openai_ok:
        print("\n✅ At least one LLM backend is working!")
    else:
        print("\n❌ No LLM backends working - check your token!")


if __name__ == "__main__":
    asyncio.run(main())
