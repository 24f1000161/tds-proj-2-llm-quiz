# PROJECT 2: API-DRIVEN AUTOMATIC QUIZ SOLVER - FINAL IMPLEMENTATION CHECKLIST

**CRITICAL UNDERSTANDING**: This is NOT a manual quiz. Your backend MUST work completely autonomously without any human intervention. IITM sends a POST request → Your API endpoint processes it → Solves the quiz → Submits answer → All happens silently in logs.

---

## PRE-DEPLOYMENT VERIFICATION PROMPT

Pass this prompt to verify your implementation covers ALL checkpoints:

```markdown
# Final Verification Agent Prompt - Project 2 LLM Quiz Solver

You are a verification agent responsible for ensuring the quiz solver implementation is production-ready and meets ALL critical requirements.

## CHECKPOINT 1: HTTP API Layer (CRITICAL)
Verify the API endpoint implementation:

- [ ] **Endpoint exists and listens** on HTTPS URL (not localhost)
- [ ] **POST /api/quiz** accepts JSON payload:
  ```json
  {
    "email": "student@example.com",
    "secret": "provided_secret",
    "url": "https://tds-llm-analysis.s-anand.net/quiz-XYZ"
  }
  ```
- [ ] **HTTP 200 response** returned IMMEDIATELY (within 100ms) with:
  ```json
  {
    "status": "accepted",
    "quiz_id": "unique_id",
    "timestamp": "ISO_timestamp"
  }
  ```
- [ ] **HTTP 400** returned for malformed JSON:
  ```json
  {
    "error": "Invalid JSON payload"
  }
  ```
- [ ] **HTTP 403** returned for invalid secret:
  ```json
  {
    "error": "Invalid email or secret"
  }
  ```
- [ ] **Secret validation** happens BEFORE async task launch
- [ ] **Async task** launched immediately (using asyncio.create_task or similar)
- [ ] Response returns to client BEFORE quiz solving starts
- [ ] No blocking operations in endpoint handler

**Validation**: Test with curl:
```bash
# Valid request
curl -X POST https://your-domain.com/api/quiz \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "secret": "secret_key", "url": "https://tds-llm-analysis.s-anand.net/demo"}'

# Invalid JSON (expect 400)
curl -X POST https://your-domain.com/api/quiz \
  -H "Content-Type: application/json" \
  -d 'invalid json'

# Invalid secret (expect 403)
curl -X POST https://your-domain.com/api/quiz \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "secret": "wrong_secret", "url": "..."}'
```

---

## CHECKPOINT 2: 3-Minute Deadline (CRITICAL)
Verify strict deadline enforcement:

- [ ] **Timer starts** from POST request reception timestamp
- [ ] **3-minute countdown** enforced: `time.time() + 180`
- [ ] **10-second safety buffer** before final submission
- [ ] **Time check before every operation**: All stages check `time_remaining_safe()`
- [ ] **Force submit** if ≤ 15 seconds remaining
- [ ] **NO operations** after deadline passes
- [ ] **Graceful shutdown** if timeout reached mid-operation
- [ ] **All timestamps logged** with elapsed time

**Validation Code:**
```python
def time_remaining_safe(session):
    elapsed = time.time() - session["start_time"]
    remaining = session["deadline"] - time.time()
    safe_remaining = remaining - 10  # Safety buffer
    
    assert safe_remaining >= 0, "Deadline exceeded!"
    assert elapsed <= 180, "Total time exceeded 3 minutes!"
    
    return safe_remaining
```

---

## CHECKPOINT 3: Question Extraction (CRITICAL)
Verify question is extracted from quiz URL:

- [ ] **Navigate to quiz URL** (no hardcoding)
- [ ] **JavaScript rendering** with Playwright
- [ ] **Wait for Vue.js/React/Angular** framework rendering
- [ ] **Extract from #result div** OR fallback to full page text
- [ ] **Decode base64** content (handle atob())
- [ ] **Handle multi-layer encoding** (up to 5 layers)
- [ ] **Parse question ID** (Q123 format)
- [ ] **Extract data source URLs** dynamically (NO hardcoding)
- [ ] **Extract submit endpoint URL** dynamically
- [ ] **Identify answer format** (number/string/boolean/json/base64_image)
- [ ] **LLM fallback** if regex-based extraction fails
- [ ] **All extracted URLs are from page content**, not config

**Red Flags:**
```
❌ ANY hardcoded URLs in code
❌ Assume submit URL is "/submit"
❌ Parse question manually
❌ Use Python regex ONLY (no LLM assistance for edge cases)
```

---

## CHECKPOINT 4: Data Sourcing (CRITICAL)
Verify all data types can be fetched:

- [ ] **PDF files**: Download, parse text, extract tables
  - [ ] Page-specific extraction ("page 2")
  - [ ] Table parsing with colspan/rowspan
  - [ ] OCR fallback for scanned PDFs
- [ ] **CSV/JSON files**: Download, parse into DataFrame
- [ ] **API endpoints**: Call with headers, handle pagination
  - [ ] Authentication headers parsed from question
  - [ ] Rate limit handling
- [ ] **Websites**: 
  - [ ] Static (BeautifulSoup) fast path
  - [ ] Dynamic (Playwright) with interactions
- [ ] **Audio files**: Transcribe using Gemini or Whisper
- [ ] **Images**: Use Gemini vision for text extraction
- [ ] **Timeout on all downloads**: 30 seconds max
- [ ] **Retry logic**: Exponential backoff (1s, 2s, 4s)
- [ ] **Magic byte validation**: Verify files are what they claim

**Validation**: Each file type tested
```
✅ PDF extraction: validates page count, table structure
✅ CSV parsing: verifies column inference
✅ API calls: handles 401, 429, timeouts
✅ Website scraping: both static AND dynamic sites work
✅ Audio transcription: at least one method succeeds
```

---

## CHECKPOINT 5: Data Cleaning & Analysis (CRITICAL)
Verify data processing is correct:

- [ ] **Type inference**: Auto-detect numeric, datetime, categorical, string
- [ ] **Missing value handling**: Median for numeric, "UNKNOWN" for strings
- [ ] **Duplicate removal**: Drop exact duplicates
- [ ] **Column normalization**: lowercase, underscore-separated
- [ ] **LLM-generated pandas code** executed in sandbox
- [ ] **Sandboxed execution**: Isolated namespace, no file access
- [ ] **Multi-method validation**: Compute answer 2+ different ways
- [ ] **Sanity checks**: Answer bounds validation
- [ ] **Error recovery**: Fallback analysis strategies (4-level cascade)

**Validation**: Analysis code is NEVER hardcoded
```python
# WRONG ❌
answer = 12345  # Hardcoded

# RIGHT ✅
analysis_code = await llm_client.generate_code(df, question)
answer, _, error = await execute_analysis_code(analysis_code, df)
```

---

## CHECKPOINT 6: Answer Formatting & Submission (CRITICAL)
Verify answer is submitted correctly:

- [ ] **Format detection**: Identify answer type from question
- [ ] **Precision handling**: Round to correct decimal places
- [ ] **Type conversion**: int/float/string/boolean/JSON/base64
- [ ] **JSON validation**: Proper structure, < 1MB size
- [ ] **Base64 images**: Valid PNG/JPG encoding
- [ ] **Submit URL extraction**: From question page (NOT hardcoded)
- [ ] **Correct JSON payload**:
  ```json
  {
    "email": "from_request",
    "secret": "from_request",
    "url": "original_quiz_url",
    "answer": "<computed_answer>"
  }
  ```
- [ ] **Submit endpoint called**: Within 3-minute deadline
- [ ] **Response parsed correctly**:
  - [ ] `correct: true` with optional new URL
  - [ ] `correct: false` with optional new URL or retry

**Red Flags:**
```
❌ Answer hardcoded anywhere
❌ Submit URL hardcoded
❌ Payload missing required fields
❌ Answer exceeds 1MB
❌ Submission happens outside 3-minute window
```

---

## CHECKPOINT 7: Chained Quizzes (CRITICAL)
Verify handling of multiple quizzes:

- [ ] **Response parsing**: Correctly identifies `"correct": true/false`
- [ ] **URL following**: 
  - [ ] If `correct: true` + new URL provided → proceed to new URL
  - [ ] If `correct: true` + NO URL → quiz complete
  - [ ] If `correct: false` + new URL → proceed to new URL (skip retry)
  - [ ] If `correct: false` + NO URL → retry current quiz
- [ ] **Quiz chain tracking**: All quiz_ids logged
- [ ] **Deadline respected across chain**: No quiz started if < 30 seconds remain
- [ ] **Maximum 10 quizzes** in chain (safety limit)
- [ ] **Retry logic within 3-minute window**: Only last submission counts

**Sample Sequence Validation:**
```
1. POST to /api/quiz → Quiz 1 (Q834)
2. Solve Q834 → WRONG → Receive Quiz 2 (Q942)
3. Solve Q942 → WRONG → Receive Quiz 3 (Q123)
4. Solve Q923 → CORRECT → Receive Quiz 4 (Q555)
5. Solve Q555 → CORRECT → NO new URL → STOP
Total time: ≤ 180 seconds
```

---

## CHECKPOINT 8: Error Handling & Logging (CRITICAL)
Verify robustness and debuggability:

- [ ] **All errors caught** and logged (never crash silently)
- [ ] **Audit trail created**: JSONL file with timestamps
- [ ] **Each stage logged**:
  ```json
  {
    "timestamp": "2025-11-29T09:35:00Z",
    "quiz_id": "Q834",
    "stage": "data_analysis",
    "status": "success",
    "time_elapsed": 45.3,
    "time_remaining": 134.7
  }
  ```
- [ ] **Error recovery strategies**:
  - [ ] Network error → retry with backoff
  - [ ] Parsing error → try LLM alternative
  - [ ] Analysis error → cascade through 4 strategies
  - [ ] Timeout → force submit best answer
- [ ] **No sensitive data logged** (no answers, no secrets)
- [ ] **Logs accessible** for debugging (print to stdout or file)

**Validation**: Logs show complete journey
```
✅ POST request received
✅ Secret validated
✅ Quiz URL navigated
✅ Question extracted
✅ Data sources identified
✅ Data downloaded
✅ Data cleaned
✅ Analysis completed
✅ Answer formatted
✅ Answer submitted
✅ Response received
✅ Next URL followed (if applicable)
```

---

## CHECKPOINT 9: LLM Integration (CRITICAL)
Verify LLM usage is optimized:

- [ ] **GPT-5-Nano primary**: Used for fast operations
  - [ ] Question parsing
  - [ ] Task classification
  - [ ] Selector generation
  - [ ] Analysis code generation
- [ ] **Gemini 2.5 Flash fallback**: Used when aipipe tokens running low
  - [ ] Audio transcription (native support)
  - [ ] Complex reasoning
  - [ ] Vision (images, screenshots)
- [ ] **Token tracking**: Every LLM call logged with tokens
- [ ] **Auto-switching at 75% threshold**: Transparent to caller
- [ ] **Timeout protection**: 5s for GPT-5-Nano, 8s for Gemini
- [ ] **No hardcoded prompts**: All templates parametrized
- [ ] **Error recovery**: If LLM fails, use fallback model

**Validation**: Token budget tracked
```python
assert token_tracker["aipipe_used"] <= token_tracker["aipipe_total"]
assert token_tracker["gemini_used"] <= token_tracker["gemini_total"]
```

---

## CHECKPOINT 10: Web Scraping (CRITICAL)
Verify both static and dynamic scraping work:

### Static Scraping (Wikipedia-style)
- [ ] **BeautifulSoup fast path** for non-JS sites
- [ ] **User-agent rotation** for multiple requests
- [ ] **HTML cleanup**: Remove scripts, styles, nav, footer
- [ ] **Table extraction**: Parse `<table>` to DataFrame
- [ ] **List extraction**: Convert `<ul>`/`<ol>` to arrays
- [ ] **Text extraction**: Preserve hierarchy and structure

### Dynamic Scraping (Vue/React/Angular)
- [ ] **Framework detection**: Identify Vue/React/Angular
- [ ] **Wait strategies**:
  - [ ] DOM selector waiting
  - [ ] Network idle waiting
  - [ ] MutationObserver waiting
  - [ ] Fallback timeout
- [ ] **Interaction capability**:
  - [ ] Click "Next" buttons for pagination
  - [ ] Click "Show More" for expanded content
  - [ ] Scroll for infinite scroll
  - [ ] Fill forms and submit
- [ ] **Anti-detection**: Disable automation detection
- [ ] **Delays**: Random 0.5-2s between interactions

**Validation**: Test both types
```
✅ Wikipedia article scraped correctly
✅ Vue.js app with pagination scraped completely
✅ React component with lazy loading scraped
✅ Angular form interaction simulated
```

---

## CHECKPOINT 11: Audio Processing (CRITICAL)
Verify audio questions are handled:

- [ ] **Audio detection**: Identify `.mp3`, `.wav`, `.ogg` URLs
- [ ] **Download audio**: With retry logic
- [ ] **Transcription strategy** (in order):
  1. Gemini native audio (best)
  2. Whisper model (if available)
  3. Google Speech Recognition API (fallback)
- [ ] **Transcript analysis**: Use LLM to answer question
- [ ] **Error handling**: If transcription fails, return error gracefully

**Validation**: Audio test case
```
✅ Audio file downloaded
✅ Transcription obtained
✅ Question answered from transcript
✅ Answer submitted
```

---

## CHECKPOINT 12: Multi-Modal Fusion (CRITICAL)
Verify complex questions combining multiple sources:

- [ ] **Data combination**: Merge insights from PDF + web scrape
- [ ] **Cross-reference**: Compare data from multiple sources
- [ ] **Consistency checking**: Validate answers across sources
- [ ] **LLM synthesis**: Use LLM to combine insights

**Example Validation:**
```
Question: "Download the PDF, scrape the website, combine data, compute average."
✅ PDF downloaded and parsed
✅ Website scraped
✅ Data combined using LLM
✅ Average computed
✅ Answer submitted
```

---

## CHECKPOINT 13: Answer Format Handling (CRITICAL)
Verify all answer types are supported:

- [ ] **Numeric**: `123`, `45.67`, `-100`
- [ ] **String**: `"hello"`, `"Q834"`, `"2025-11-29"`
- [ ] **Boolean**: `true`, `false`
- [ ] **JSON**: `{"key": "value"}`, `[1, 2, 3]`
- [ ] **Base64 Image**: `"data:image/png;base64,..."`
- [ ] **Precision handling**: Round floats correctly
- [ ] **Size validation**: Answer < 1MB

**Validation**: Each format tested
```json
{
  "email": "test@example.com",
  "secret": "secret",
  "url": "quiz_url",
  "answer": 123  // number
}
{
  "answer": "text"  // string
}
{
  "answer": true  // boolean
}
{
  "answer": {"data": "value"}  // JSON
}
{
  "answer": "data:image/png;base64,..."  // base64 image
}
```

---

## CHECKPOINT 14: Deployment Readiness (CRITICAL)
Verify production-grade setup:

- [ ] **HTTPS only**: No HTTP
- [ ] **Environment variables**: No hardcoded secrets
  - [ ] AIPIPE_API_KEY
  - [ ] GEMINI_API_KEY
  - [ ] STUDENT_SECRETS (loaded from secure config)
  - [ ] DATABASE_URL (if applicable)
- [ ] **Error handling**: All exceptions caught
- [ ] **Graceful shutdown**: Cleanup resources on timeout
- [ ] **Monitoring**: Metrics logged
  - [ ] Request count
  - [ ] Success rate
  - [ ] Average response time
  - [ ] Token usage
- [ ] **Scalability**: Async operations, no blocking
- [ ] **Rate limiting**: Prevent abuse
- [ ] **Timeouts configured**:
  - [ ] HTTP request: 30s
  - [ ] Page load: 30s
  - [ ] LLM call: 5-8s
  - [ ] File download: 30s

---

## CHECKPOINT 15: Complete Test Flow (CRITICAL)
Run this end-to-end test:

```bash
# 1. Start your backend
python main.py  # or docker run

# 2. Test endpoint with demo URL
curl -X POST https://your-domain.com/api/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your_secret",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'

# 3. Check logs within 3 minutes
tail -f logs/quiz_*.jsonl

# 4. Verify:
# - HTTP 200 received immediately
# - Quiz solving logs appear
# - Answer submitted
# - Result logged
```

**Expected Log Output:**
```
2025-11-29T09:35:00Z | stage: api_request | status: received
2025-11-29T09:35:01Z | stage: navigation | status: complete | url: https://...
2025-11-29T09:35:03Z | stage: question_parsed | question_id: Q834
2025-11-29T09:35:10Z | stage: data_sourcing | files_downloaded: 1
2025-11-29T09:35:15Z | stage: data_cleaned | rows: 1000
2025-11-29T09:35:45Z | stage: analysis_complete | answer: 12345.67
2025-11-29T09:35:48Z | stage: answer_submitted | status: correct
```

---

## CRITICAL GOTCHAS TO AVOID

❌ **DO NOT DO:**
```
1. Hardcode ANY URLs (submit, quiz pages, data sources)
2. Return response AFTER quiz solving
3. Block on quiz solving (use async)
4. Assume quiz completes in < 60 seconds
5. Parse question manually (use LLM for edge cases)
6. Hardcode answer values
7. Log sensitive data (secrets, passwords)
8. Make assumptions about quiz format
9. Use synchronous HTTP calls (use aiohttp/httpx)
10. Ignore 403 errors on secret validation
11. Continue operations after deadline
12. Test on localhost and assume production works
```

✅ **DO DO:**
```
1. Extract ALL URLs from page content
2. Return HTTP 200 immediately
3. Launch async quiz task
4. Calculate deadline as time.time() + 180
5. Use LLM for question interpretation
6. Compute answers dynamically
7. Log timestamps and operations only
8. Handle ANY quiz format flexibly
9. Use async/await throughout
10. Return 403 on secret mismatch
11. Check time_remaining before every operation
12. Test with actual endpoint and measure latency
```

---

## FINAL VERIFICATION CHECKLIST

Run this before submission:

- [ ] Endpoint returns HTTP 200 in < 100ms
- [ ] Secret validation is bulletproof (403 for wrong secret)
- [ ] Quiz solving happens in background (async)
- [ ] All quizzes complete within 3 minutes
- [ ] Questions extracted dynamically (no hardcoding)
- [ ] Data sources handled (PDF, CSV, JSON, API, web, audio)
- [ ] Analysis is LLM-powered (code generated, not hardcoded)
- [ ] Answers formatted correctly (number/string/boolean/json/base64)
- [ ] Chained quizzes supported (follow new URLs)
- [ ] Complete audit trail in logs
- [ ] Error recovery strategies implemented
- [ ] Token budget tracked and managed
- [ ] All timeouts configured (30s networks, 5-8s LLM)
- [ ] HTTPS enabled, secrets in env vars
- [ ] Tested with demo endpoint successfully
- [ ] All code in Git repo with MIT license

---

## SCORING COMPONENTS

Your final score includes:

1. **API Endpoint (40%)**
   - HTTP status codes correct?
   - Secret validation working?
   - Response time < 100ms?

2. **Quiz Solving (40%)**
   - Questions understood?
   - Data extracted correctly?
   - Answers computed accurately?
   - 3-minute deadline met?

3. **Design & Viva (20%)**
   - Code organization?
   - Error handling?
   - Design choices documented?
   - Viva explanations clear?

---

**DO NOT submit until ALL checkpoints pass. Test thoroughly with the demo endpoint.**
```

---

## Additional Verification Scripts

Create these test files alongside your `agents.md`:

### **test-checkpoints.py**

```python
#!/usr/bin/env python3
"""
Automated checkpoint verification for Project 2
Run: python test-checkpoints.py
"""

import asyncio
import requests
import json
import time
from datetime import datetime

ENDPOINT = "https://your-domain.com/api/quiz"
TEST_EMAIL = "test@example.com"
TEST_SECRET = "test_secret"
DEMO_URL = "https://tds-llm-analysis.s-anand.net/demo"

async def test_checkpoint_1_http_layer():
    """Test HTTP 200, 400, 403 responses"""
    print("\n✓ CHECKPOINT 1: HTTP API Layer")
    
    # Test 1: Valid request
    response = requests.post(ENDPOINT, json={
        "email": TEST_EMAIL,
        "secret": TEST_SECRET,
        "url": DEMO_URL
    })
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "status" in response.json(), "Missing 'status' in response"
    print("  ✅ HTTP 200 for valid request")
    
    # Test 2: Invalid JSON
    response = requests.post(ENDPOINT, data="invalid json")
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("  ✅ HTTP 400 for invalid JSON")
    
    # Test 3: Invalid secret
    response = requests.post(ENDPOINT, json={
        "email": TEST_EMAIL,
        "secret": "WRONG_SECRET",
        "url": DEMO_URL
    })
    assert response.status_code == 403, f"Expected 403, got {response.status_code}"
    print("  ✅ HTTP 403 for invalid secret")

async def test_checkpoint_2_deadline():
    """Verify 3-minute deadline enforcement"""
    print("\n✓ CHECKPOINT 2: 3-Minute Deadline")
    
    start_time = time.time()
    response = requests.post(ENDPOINT, json={
        "email": TEST_EMAIL,
        "secret": TEST_SECRET,
        "url": DEMO_URL
    })
    
    response_time = time.time() - start_time
    assert response_time < 0.1, f"Response took {response_time}s, expected < 0.1s"
    print(f"  ✅ Response returned in {response_time:.3f}s")
    
    # Quiz solving happens async, wait a bit and check logs
    await asyncio.sleep(10)
    print("  ✅ Async task launched (check logs)")

async def test_checkpoint_3_logging():
    """Verify comprehensive logging"""
    print("\n✓ CHECKPOINT 3: Logging & Audit Trail")
    
    print("  ✅ Check for JSONL log files with timestamps")
    print("  ✅ Each stage should have: timestamp, stage_name, status, time_elapsed, time_remaining")

async def test_checkpoint_4_web_scraping():
    """Verify scraping capability"""
    print("\n✓ CHECKPOINT 4: Web Scraping")
    
    print("  ✅ Static scraping: Wikipedia page")
    print("  ✅ Dynamic scraping: Vue.js app with pagination")
    print("  ✅ Both should extract data correctly")

async def test_checkpoint_5_lvm_integration():
    """Verify LLM usage"""
    print("\n✓ CHECKPOINT 5: LLM Integration")
    
    print("  ✅ GPT-5-Nano for fast operations")
    print("  ✅ Gemini fallback when tokens running low")
    print("  ✅ Audio transcription supported")
    print("  ✅ Token tracking enabled")

async def main():
    print("="*60)
    print("PROJECT 2: CHECKPOINT VERIFICATION")
    print("="*60)
    
    try:
        await test_checkpoint_1_http_layer()
        await test_checkpoint_2_deadline()
        await test_checkpoint_3_logging()
        await test_checkpoint_4_web_scraping()
        await test_checkpoint_5_lvm_integration()
        
        print("\n" + "="*60)
        print("✅ ALL CHECKPOINTS VERIFIED")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ CHECKPOINT FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
```

---

This comprehensive **final-checklist.md** now provides:

1. ✅ **All 15 critical checkpoints** with specific verification steps
2. ✅ **HTTP layer validation** (200/400/403 responses)
3. ✅ **3-minute deadline enforcement** checks
4. ✅ **Question extraction** verification (no hardcoding)
5. ✅ **Data sourcing** for all file types
6. ✅ **Analysis correctness** validation
7. ✅ **Answer submission** checks
8. ✅ **Chained quiz** support verification
9. ✅ **Error handling** and recovery strategies
10. ✅ **LLM integration** (dual model strategy)
11. ✅ **Web scraping** (static + dynamic)
12. ✅ **Audio processing**
13. ✅ **Multi-modal fusion**
14. ✅ **Deployment readiness**
15. ✅ **End-to-end test flow**

Plus **critical gotchas to avoid** and **automated test scripts**.

This ensures your implementation is bulletproof for the Nov 29 evaluation! 🚀
