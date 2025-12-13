# QUICK FIX CHECKLIST - Get Under 180s Deadline

## 🎯 Target: Reduce Quiz 2 from 199s → 110s (90 seconds saved)

---

## ✅ FIX #1: Remove Duplicate Classification (38s saved)

**File:** `quiz_solver/task_handlers.py`

**Find this in `handle_command_task()`:**
```python
# ❌ DELETE THESE LINES:
task_analysis = await llm_client.classify_task_dynamically(question, context)
classification_result = await llm_client.classify_task(question)
task_spec = await llm_client.analyze_question_type(question)
```

**Why:** The main pipeline already called `classify_task_dynamically()`. The handler received the result in the `classification` parameter. Don't call it again!

**Verify:**
- [ ] Removed all `classify_task_dynamically()` calls from task_handlers.py
- [ ] Removed all `classify_task()` calls from task_handlers.py
- [ ] Removed all `analyze_question_type()` calls from task_handlers.py
- [ ] Handler now uses the `classification` parameter passed in

**Time saved:** 38 seconds

---

## ✅ FIX #2: Combine Sequential LLM Calls (20-30s saved)

**File:** `quiz_solver/llm_client.py`

**Add this method:**
```python
async def generate_complete_command(self, question: str) -> str:
    """
    Generate COMPLETE command in ONE LLM call.
    Includes: URL extraction, command building, and formatting.
    """
    prompt = f"""Based on this question, generate the EXACT command string to submit.
    
Question: {question}

Requirements:
- Include all necessary flags and headers
- Use the email from the URL if personalization needed
- Return ONLY the command, no explanation
- Command must be ready to submit as-is"""

    response = await self.call_aipipe(
        system="Generate exact, working commands with all details included.",
        messages=[{"role": "user", "content": prompt}],
        timeout=60  # Allow up to 60s for complex commands
    )
    
    return response.strip()
```

**Update `handle_command_task()`:**
```python
async def handle_command_task(question, context, classification, session):
    # ✅ NEW: Single optimized call
    command = await llm_client.generate_complete_command(question)
    return command
```

**Verify:**
- [ ] New method added to llm_client.py
- [ ] handle_command_task() calls only generate_complete_command()
- [ ] No separate URL extraction calls
- [ ] No separate formatting calls
- [ ] Method works correctly for different command types (uv, git, curl, etc.)

**Time saved:** 20-30 seconds

---

## ✅ FIX #3: Remove Redundant Post-Generation Analysis (54s saved)

**File:** `quiz_solver/llm_analysis.py`

**Find in `solve_with_llm()`:**
```python
# ❌ DELETE THIS BLOCK:
answer = await handler(question, context, classification, session)
analysis = await llm_client.analyze_strategy(answer, question)
final_answer = await llm_client.extract_answer_from_analysis(analysis)

# ✅ REPLACE WITH:
answer = await handler(question, context, classification, session)
# Handler already returns the final, formatted answer
return answer  # No more processing needed
```

**Why:** The handler's job is to return the complete answer. Don't re-analyze it afterward.

**Verify:**
- [ ] Removed `analyze_strategy()` calls
- [ ] Removed `extract_answer_from_analysis()` calls
- [ ] Removed `process_answer()` calls
- [ ] Handler returns are used directly without post-processing
- [ ] Tests pass with simplified answer processing

**Time saved:** 54 seconds

---

## ✅ FIX #4: Add Timeouts to All LLM Calls (Safety)

**File:** `quiz_solver/llm_client.py`

**Update all `call_aipipe()` calls:**
```python
# Every call should have a timeout
async def call_aipipe(self, system: str, messages: list, timeout: int = 30):
    """
    timeout: Maximum seconds to wait (default 30s)
    """
    try:
        response = await asyncio.wait_for(
            self._call_aipipe_impl(system, messages),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {timeout}s, using fallback")
        return "fallback_answer"  # or retry logic
```

**Apply to key methods:**
```python
# In classify_task_dynamically():
await self.call_aipipe(..., timeout=40)  # 40s for complex classification

# In generate_complete_command():
await self.call_aipipe(..., timeout=60)  # 60s for command generation

# In other methods:
await self.call_aipipe(..., timeout=30)  # 30s default
```

**Verify:**
- [ ] All LLM calls have explicit timeout parameter
- [ ] Timeouts are realistic (not too short, not too long)
- [ ] Timeout exceeded is logged and handled gracefully
- [ ] No hanging requests can consume all remaining time

**Time saved:** Prevents runaway calls eating deadline

---

## ✅ FIX #5: Parallelize Data Fetching (20-40s saved)

**File:** `quiz_solver/datasourcing.py`

**Find sequential data fetching:**
```python
# ❌ BEFORE: Sequential (slow)
data1 = await fetch_github_api(...)  # 20s
data2 = await fetch_webpage(...)     # 20s  
data3 = await fetch_csv_file(...)    # 20s
# Total: 60s

# ✅ AFTER: Parallel (fast)
data1, data2, data3 = await asyncio.gather(
    fetch_github_api(...),
    fetch_webpage(...),
    fetch_csv_file(...),
    return_exceptions=True  # Continue if one fails
)
# Total: 20s (same time as slowest call)
```

**Verify:**
- [ ] Data source fetching uses `asyncio.gather()`
- [ ] Independent requests happen in parallel
- [ ] Error handling doesn't block other requests
- [ ] Tests pass with parallel fetching

**Time saved:** 20-40 seconds

---

## 📋 Verification Checklist

After each fix, verify:

- [ ] Code compiles without errors
- [ ] Tests run without hanging
- [ ] No LLM calls are duplicated
- [ ] Timeline shows reduced stage durations
- [ ] Quiz 2 completes in < 150 seconds
- [ ] All answers are still correct

---

## 🧪 Testing Script

```python
# test_performance.py
import time
import asyncio

async def test_quiz_2_performance():
    """Test that Quiz 2 completes in time."""
    
    # Simulate Quiz 2 solving
    start = time.time()
    
    # Your pipeline
    answer = await solve_quiz_pipeline(
        email="test@example.com",
        secret="secret",
        url="https://tds-llm-analysis.s-anand.net/project2-uv",
        deadline=time.time() + 180
    )
    
    elapsed = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"Quiz 2 Performance Test")
    print(f"{'='*60}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Deadline:    180.0s")
    print(f"Status:      {'✅ PASS' if elapsed < 180 else '❌ FAIL'}")
    print(f"Margin:      {180 - elapsed:.1f}s")
    print(f"{'='*60}\n")
    
    assert elapsed < 180, f"Exceeded deadline by {elapsed - 180:.1f}s"
    assert answer is not None, "No answer generated"
    
    return True

# Run test
asyncio.run(test_quiz_2_performance())
```

---

## 📊 Expected Timeline After Fixes

```
BEFORE (Current - FAILS):
11:48:57 - Quiz starts
11:49:42 - Classification (45s) ← LLM CALL 1
11:50:20 - Task classification (38s) ← LLM CALL 2 REMOVED ❌
11:51:22 - Command gen (62s) ← LLM CALL 3 OPTIMIZED
11:51:57 - Analysis (35s) ← LLM CALL 4 REMOVED ❌
11:52:16 - Answer (19s) ← LLM CALL 5 REMOVED ❌
11:52:18 - Submit (2s)
TOTAL: 199s ❌ EXCEEDS BY 19s

AFTER (Fixed - PASSES):
11:48:57 - Quiz starts
11:49:42 - Classification (45s) ← LLM CALL 1
11:50:45 - Command generation (63s) ← LLM CALL 2 (combined, optimized)
11:51:48 - Submission (2s)
TOTAL: 111s ✅ SAFE MARGIN (69s buffer)
```

---

## 🚀 Quick Reference: Files to Modify

| File | Changes | Time |
|------|---------|------|
| `task_handlers.py` | Remove duplicate classifications | 5 min |
| `llm_client.py` | Add `generate_complete_command()` + timeouts | 15 min |
| `llm_analysis.py` | Remove post-generation analysis | 5 min |
| `datasourcing.py` | Use `asyncio.gather()` for parallel fetching | 10 min |

**Total implementation time:** ~35 minutes
**Time saved:** ~90 seconds per quiz
**Impact:** From failing to passing all quizzes

---

## ✅ Final Validation

Before submitting, verify:

```python
# In your logs, you should see:
✓ Task classified: command_generation (no duplicate classification)
✓ Generated command: uv http get ... (single combined call)
✓ Final answer: ... (no post-generation analysis)
⏱️ Time used for this quiz: 110-120s (not 199s)
✅ CORRECT! (submission succeeds)
```

---

## 📞 If Still Slow

If you're still over 150s per quiz, check:

1. **Are there still duplicate LLM calls?**
   - Search code for `classify_task_dynamically()` - should only appear ONCE in pipeline
   - Search for `await llm_client` - count total calls per quiz

2. **Are data sources being fetched sequentially?**
   - Should use `asyncio.gather()` for parallel fetching
   - Individual fetch times visible in logs should be much faster

3. **Are timeouts set too high?**
   - Default 30s is fine for most calls
   - Command generation 60s max
   - Classification 40s max
   - If any call exceeds these, log it

4. **Is browser performance an issue?**
   - Navigation: ~5s (OK)
   - Content extraction: ~8s (OK)
   - If either is >15s, there's a problem

Good luck! You've got this. 🚀
