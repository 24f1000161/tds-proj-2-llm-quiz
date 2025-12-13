# AGENTS.MD - Complete Instructions for Next Developer

## Context: You're Taking Over a Quiz-Solving LLM System

**Current Status:**
- Success Rate: 50% (2/4 quizzes passing)
- Architecture: Dynamic LLM-driven (NOT hardcoded - this is good)
- Primary Problem: 30-second timeout too short (67% of failures)
- Secondary Problems: Missing infrastructure handlers (33% of failures)

**Key Discovery:** Your predecessor correctly identified that timeouts are the bottleneck. They consume 75% of the deadline on complex questions.

---

## Part 1: The Problem (What You're Fixing)

### Test Results Summary

| Quiz | Type | Status | Time | Root Cause |
|------|------|--------|------|-----------|
| 1 | GitHub API | ✅ PASS | 56s | None |
| 2 | Logs ZIP | ❌ FAIL | 50s | No ZIP→DataFrame converter |
| 3 | PDF Invoice | ✅ PASS | 77s | None |
| 4 | CSV Orders | ❌ TIMEOUT | 235s | **timeout=30 is too short** |

### Why Quiz 4 Fails: The Timeout Issue

```
Timeline (180 second deadline):
├─ 12:22:06 - Start classification
├─ 12:22:36 - TIMEOUT #1 (30s wasted)
├─ 12:23:06 - TIMEOUT #2 (30s wasted)
├─ 12:23:32 - Finally succeeds after 120s
├─ 12:23:53 - Start analysis
├─ 12:24:23 - TIMEOUT #1 (30s wasted)
├─ 12:24:53 - TIMEOUT #2 (30s wasted)
└─ 12:25:22 - Code error
Result: EXCEEDS DEADLINE (185s > 180s)

Time Analysis:
├─ Classification timeouts: 120s (67% of deadline)
├─ Analysis timeouts: 60s (33% of deadline)
├─ Actual work time: 0s
└─ Total wasted: 180+ seconds
```

**Why it happens:**
- Complex questions take 25-30 seconds to analyze (LLM processing time)
- Timeout set at 30 seconds (too tight margin)
- When processing hits 30s, timeout fires
- System retries up to 3 times before giving up
- Total: 120+ seconds wasted on classification alone

---

## Part 2: Root Causes (Ranked by Impact)

### 🔴 PRIMARY: Configuration - 30-Second Timeout [CRITICAL]

**File:** `llm_client.py` or equivalent LLM client

**What to Look For:**
```python
async with asyncio.timeout(30):  # ← TOO SHORT
    response = await session.post(...)

# OR

timeout=30  # ← TOO SHORT
```

**Why It's Wrong:**
- Simple questions: 5-10s (OK)
- Moderate questions: 15-20s (OK)
- Complex questions: 25-30s (HITS TIMEOUT!)

**The Fix:**
```python
async with asyncio.timeout(45):  # ← INCREASED
    response = await session.post(...)

# OR

timeout=45  # ← INCREASED
```

**Impact:** Saves 120+ seconds on Quiz 4
**Effort:** 1 minute

---

### 🟠 SECONDARY: Bloated Classification Prompt [HIGH]

**File:** `task_classification.py` or wherever `classify_task_dynamically()` is defined

**What to Look For:**
```python
async def classify_task_dynamically(question, context):
    system_prompt = """
    You are analyzing a quiz question...
    Identify the task type...
    Extract data sources...
    Determine data formats...
    Plan the analysis approach...
    List processing steps...
    [20+ lines of detailed instructions]
    """  # ← 500+ TOKENS (WAY TOO MUCH)
```

**Why It's Wrong:**
- Asks LLM for 20+ different pieces of information
- Each piece requires processing time
- Total processing: 15-30 seconds per question
- At 30-second timeout, it just barely fits (and sometimes exceeds)

**The Fix:**
```python
async def classify_task_dynamically(question, context):
    system_prompt = """
    Classify this quiz question into ONE category only:
    - api_call
    - data_analysis
    - command_generation
    - other
    
    Respond with ONLY the category name. No explanation.
    """  # ← 50 TOKENS (10X SMALLER)
```

**Impact:** Classification completes in 5-10 seconds instead of 20-30
**Effort:** 15 minutes

---

### 🟡 TERTIARY: Missing Retry Backoff [MEDIUM]

**File:** `llm_client.py` or wherever retry logic happens

**What to Look For:**
```python
except asyncio.TimeoutError:
    if attempt < 4:
        await asyncio.sleep(5)  # ← HARDCODED
        # Retry
```

**Why It's Wrong:**
- Uses fixed 5-second wait between retries
- Doesn't give LLM time to recover
- Retries immediately with same parameters
- Higher chance of hitting same timeout again

**The Fix:**
```python
except asyncio.TimeoutError:
    if attempt < 4:
        wait_time = min(5 + attempt * 2, 15)  # Progressive: 5s, 7s, 9s, 15s
        await asyncio.sleep(wait_time)
        # Retry
```

**Impact:** Better recovery rate on timeouts
**Effort:** 5 minutes

---

### 🔴 SECONDARY: Missing ZIP Handler [CRITICAL]

**File:** `datasourcing.py` or data sourcing module

**What to Look For:**
```python
# Extracts ZIP successfully
print("Extracted ZIP: 1 files, 4 log entries")

# But then in data preparation:
print("No structured DataFrame available")  # ← GAP HERE
```

**Why It's Wrong:**
- ZIP file is extracted but not converted to DataFrame
- Analysis handler expects DataFrame, gets nothing
- Falls back to generic "start" answer (wrong)

**The Fix:**
```python
# After extracting ZIP entries:
if zip_entries:
    import pandas as pd
    df = pd.DataFrame(zip_entries)  # Convert to DataFrame
    context['dataframe'] = df  # Store for analysis
    return df
```

**Impact:** Quiz 2 returns correct answer instead of "start"
**Effort:** 30 minutes

---

### 🔴 SECONDARY: Missing Sandbox Imports [CRITICAL]

**File:** `execute_analysis_code()` or code execution function

**What to Look For:**
```python
def execute_analysis_code(code_str, df=None):
    # Missing imports! Just runs code directly
    exec(code_str)  # ← FAILS with "requests not defined"
```

**Why It's Wrong:**
- LLM-generated code uses 'requests', 'pandas', etc.
- These modules aren't imported in execution namespace
- Code execution fails with NameError

**The Fix:**
```python
def execute_analysis_code(code_str, df=None):
    # Setup all necessary imports FIRST
    import pandas as pd
    import numpy as np
    import json
    import requests
    import re
    import datetime
    
    namespace = {
        'pd': pd,
        'np': np,
        'json': json,
        'requests': requests,
        're': re,
        'datetime': datetime,
    }
    
    if df is not None:
        namespace['df'] = df
    
    try:
        exec(code_str, namespace)
        return namespace.get('answer')
    except Exception as e:
        raise RuntimeError(f"Code execution failed: {e}")
```

**Impact:** Code execution works instead of failing
**Effort:** 5 minutes

---

### 🟠 SECONDARY: Generic Fallback Answer [HIGH]

**File:** `llm_analysis.py` or answer generation logic

**What to Look For:**
```python
if not answer_generated:
    answer = "start"  # ← GENERIC, WRONG FOR MOST QUESTIONS
```

**Why It's Wrong:**
- "start" is only correct for the intro quiz
- All other questions have different answers
- When analysis fails, system submits wrong answer
- Better to fail explicitly

**The Fix:**
```python
if not answer_generated:
    raise Exception(f"Failed to generate answer for {task_type}")
    # OR use task-specific fallbacks:
    # Don't have a sensible fallback for most tasks
    # Better to fail and log error
```

**Impact:** No more wrong "start" answers
**Effort:** 5 minutes

---

### 🟡 SECONDARY: No DataFrame Validation [MEDIUM]

**File:** `handle_data_analysis_task()` in task handlers

**What to Look For:**
```python
async def handle_data_analysis_task(question, context):
    df = context.get('dataframe')
    
    if df is None:
        return None  # ← SILENT FAILURE
```

**Why It's Wrong:**
- When DataFrame doesn't exist, returns None silently
- Doesn't try to convert available data
- Doesn't provide helpful error message
- Falls back to generic answer

**The Fix:**
```python
async def handle_data_analysis_task(question, context):
    df = context.get('dataframe')
    
    # Try to create DataFrame if data exists but not converted
    if df is None and 'raw_data' in context:
        df = pd.DataFrame(context['raw_data'])
    
    if df is None:
        raise Exception(
            f"No DataFrame available. "
            f"Available context: {list(context.keys())}"
        )
    
    # Now proceed with analysis
    answer = await generate_analysis(df, question)
    return answer
```

**Impact:** Better diagnostics, automatic conversion
**Effort:** 10 minutes

---

## Part 3: Priority & Implementation Order

### CRITICAL PATH (Do These First - 20 minutes total)

These fixes unblock everything else by solving the timeout problem.

**1. Increase Timeout (1 minute)**
```
File: llm_client.py
Change: timeout=30 → timeout=45
Impact: Huge - prevents timeouts
```

**2. Simplify Prompt (15 minutes)**
```
File: task_classification.py
Change: 500-token detailed prompt → 50-token simple prompt
Impact: 10x faster classification
```

**3. Progressive Backoff (5 minutes)**
```
File: llm_client.py
Change: Fixed sleep(5) → Progressive sleep(5+attempt*2)
Impact: Better retry recovery
```

**Result after these 3 fixes:**
- Quiz 4 time: 235s → 50s ✅
- No more timeout warnings ✅
- Deadline no longer exceeded ✅

---

### SECONDARY PATH (Do These Next - 45 minutes total)

These fixes address infrastructure gaps and enable 100% pass rate.

**4. Add Sandbox Imports (5 minutes)**
```
File: Code execution function
Add: pandas, numpy, requests, json, re, datetime to namespace
Impact: Code execution works
```

**5. ZIP→DataFrame Converter (30 minutes)**
```
File: datasourcing.py
Add: df = pd.DataFrame(extracted_entries)
Impact: Quiz 2 returns correct number
```

**6. Better Error Handling (5 minutes)**
```
File: llm_analysis.py
Remove: Generic "start" fallback
Add: Explicit exception raising
Impact: No wrong answers
```

**7. DataFrame Validation (10 minutes)**
```
File: task_handlers.py
Add: Check and convert data before analysis
Impact: Better diagnostics
```

**Result after all fixes:**
- All 4 quizzes passing ✅
- Success rate: 100% ✅
- Time budget: Well under deadline ✅

---

## Part 4: Testing Strategy

### After Timeout Fixes (Verify #1-3)

```bash
# Test that classification is faster
python test_quiz_4.py --measure-time

Expected BEFORE: 235 seconds ❌
Expected AFTER: 50 seconds ✅
Expected change: 180+ seconds saved!
```

### After Infrastructure Fixes (Verify #4-7)

```bash
# Test individual quizzes
python test_quiz_1.py  # Should still pass
python test_quiz_2.py  # Should now return number instead of "start"
python test_quiz_3.py  # Should still pass
python test_quiz_4.py  # Should now return JSON instead of "start"

# Test all together
python test_all_quizzes.py

Expected: 4/4 passing ✅
Expected success rate: 100% ✅
```

### Specific Log Indicators

**Good signs (after timeout fix):**
```
✅ No "LLM call timed out after 30s" messages
✅ Classification completes in 8-10 seconds
✅ Quiz 4 total time under 180 seconds
```

**Good signs (after infrastructure fix):**
```
✅ Quiz 2 returns a number (not "start")
✅ Quiz 4 returns JSON (not "start")
✅ No "requests not defined" errors
✅ No "No DataFrame available" errors
```

---

## Part 5: Code Locations Reference

| Issue | File | Function | Look For | Change |
|-------|------|----------|----------|--------|
| **Timeout too short** | llm_client.py | call_aipipe_with_retries | `timeout=30` | Change to `45` |
| **Bloated prompt** | task_classification.py | classify_task_dynamically | `system_prompt = """..."""` | Simplify to one sentence |
| **Retry backoff** | llm_client.py | retry logic | `sleep(5)` | Use progressive: `5+attempt*2` |
| **Missing imports** | execute_analysis_code | code execution setup | No imports in namespace | Add all stdlib + 3rd party |
| **ZIP handling** | datasourcing.py | After ZIP extraction | Raw entries, no DataFrame | Convert with `pd.DataFrame()` |
| **Generic fallback** | llm_analysis.py | Answer generation | `fallback = "start"` | Remove or use `raise Exception` |
| **DataFrame check** | task_handlers.py | handle_data_analysis_task | `if df is None: return None` | Check & convert or raise error |

---

## Part 6: Success Checklist

Before marking as complete, verify:

- [ ] Found and increased `timeout=30` → `timeout=45`
- [ ] Simplified classification prompt (now <100 tokens)
- [ ] Added progressive backoff to retry logic
- [ ] **Test:** Quiz 4 time drops from 235s to ~50s
- [ ] Added imports to sandbox (pandas, numpy, requests, json, re, datetime)
- [ ] **Test:** Code execution no longer has "requests not defined" error
- [ ] Added ZIP→DataFrame converter
- [ ] **Test:** Quiz 2 returns a number, not "start"
- [ ] Removed/fixed generic fallback answer
- [ ] **Test:** Quiz 4 returns JSON, not "start"
- [ ] Added DataFrame validation
- [ ] Quiz 1 still passes (56s)
- [ ] Quiz 2 now passes (40s)
- [ ] Quiz 3 still passes (77s)
- [ ] Quiz 4 now passes (50s)
- [ ] No timeout warnings in logs
- [ ] All tests pass: 4/4 ✅
- [ ] Total time under deadline ✅

---

## Part 7: Expected Results

### BEFORE

```
Quiz 1: ✅ PASS (56s)
Quiz 2: ❌ FAIL - "start" (wrong)
Quiz 3: ✅ PASS (77s)
Quiz 4: ❌ FAIL - TIMEOUT (235s, exceeds deadline)

Success Rate: 50%
Total time: 236+ seconds (exceeds 180s per quiz)
```

### AFTER TIMEOUT FIX (20 minutes)

```
Quiz 1: ✅ PASS (56s)
Quiz 2: ❌ FAIL - "start" (infrastructure issue remains)
Quiz 3: ✅ PASS (77s)
Quiz 4: ⚠️  PARTIAL (50s, no longer timeout, but wrong answer due to missing imports)

Success Rate: 25-50%
Total time: 150s (UNDER DEADLINE) ✅
```

### AFTER ALL FIXES (66 minutes)

```
Quiz 1: ✅ PASS (56s)
Quiz 2: ✅ PASS (40s)
Quiz 3: ✅ PASS (77s)
Quiz 4: ✅ PASS (50s)

Success Rate: 100%
Total time: 223s out of 720s available
Time remaining: 497s for additional quizzes
```

---

## Part 8: Architecture Overview (What You're Working With)

### System Flow

```
┌─────────────────────────────────────────────────────┐
│ Input: Quiz Question + 180s Deadline                │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 1: Navigation (2s)                            │
│ └─ Load quiz page                                   │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 2: Content Extraction (3s)                    │
│ └─ Extract question text and links                  │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 3: Question Parsing (1s)                      │
│ └─ Identify data sources and submit URL             │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 4: LLM Classification ⚠️  (TIMEOUT ISSUE!)   │
│ ├─ Call Aipipe/Gemini with question                │
│ ├─ timeout=30 (TOO SHORT!)                         │
│ ├─ For complex Q: Timeout fires, retries 3×        │
│ └─ Result: 120s wasted                             │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 5: Data Sourcing (5s)                        │
│ ├─ Download CSV/ZIP/PDF                            │
│ └─ Extract data (sometimes to DataFrame, sometimes not)
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 6: Data Preparation (varies)                 │
│ ├─ Convert to DataFrame (if handler exists)        │
│ └─ ⚠️  Missing ZIP handler here                     │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 7: LLM Analysis (15-30s)                      │
│ ├─ Generate analysis code/steps                     │
│ └─ ⚠️  Also can timeout if processing is slow       │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 8: Code Execution (optional, 5-10s)         │
│ ├─ Execute generated analysis code                 │
│ └─ ⚠️  Missing imports cause failures               │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Stage 9: Answer Submission (1s)                    │
│ └─ POST answer to quiz endpoint                    │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Output: Success/Failure + Next URL                 │
└─────────────────────────────────────────────────────┘
```

**Problem Areas:**
- ⚠️  Stage 4: Timeout=30 is too short (120s wasted)
- ⚠️  Stage 6: No ZIP handler (Quiz 2 fails)
- ⚠️  Stage 7: Bloated prompt (adds 10-15s unnecessarily)
- ⚠️  Stage 8: Missing imports (code execution fails)

---

## Part 9: Quick Reference

### One-Line Summaries

| Issue | Fix | Time | Impact |
|-------|-----|------|--------|
| Timeout too short | Change 30→45 | 1 min | Saves 120s |
| Bloated prompt | Simplify to 50 tokens | 15 min | 10x faster |
| No retry backoff | Add progressive sleep | 5 min | Better recovery |
| No ZIP handler | Add DataFrame conversion | 30 min | Quiz 2 works |
| Missing imports | Add to sandbox | 5 min | Code executes |
| Bad fallback | Remove "start" | 5 min | No wrong answers |
| No DataFrame check | Validate before analysis | 10 min | Better diagnostics |

---

## Part 10: Common Pitfalls & How to Avoid Them

### ❌ Don't

- Don't revert to hardcoded patterns (you'd undo good work)
- Don't increase timeout to 120s (that masks the real problem)
- Don't skip the prompt simplification (reduces timeout pressure)
- Don't forget to test after each fix (you won't know if it worked)
- Don't increase timeouts without simplifying prompts (wastes time)

### ✅ Do

- Do fix in priority order (timeouts first, infrastructure second)
- Do test after each fix group (verify before moving on)
- Do check logs for timeout warnings (key indicator)
- Do measure time before/after (prove the improvement)
- Do simplify prompts AND increase timeouts (double solution)

---

## Final Summary

### Your Mission

Fix a 50% pass rate system that's timing out on complex questions.

### Root Cause

30-second timeout is too short for complex LLM analysis (25-30s processing time).

### Solution Priority

1. **Timeouts (20 min):** Biggest impact, quickest fix
   - Increase timeout to 45s
   - Simplify classification prompt
   - Add retry backoff

2. **Infrastructure (45 min):** Fixes remaining errors
   - Add sandbox imports
   - ZIP→DataFrame converter
   - Better error handling
   - DataFrame validation

### Expected Outcome

- ✅ All 4 quizzes passing
- ✅ 100% success rate
- ✅ Under deadline for all quizzes
- ✅ No timeout warnings
- ✅ Clean error messages

### Effort

- Timeout fixes: 20 minutes (huge payoff)
- Infrastructure fixes: 45 minutes (completes solution)
- Testing: 15 minutes (validation)
- **Total: 66-80 minutes**

### Good Luck!

You've got clear instructions, specific file locations, and expected results.
The previous developer did the hard part (removing hardcoded patterns).
You just need to optimize configuration and add missing pieces.

You've got this! 🚀
