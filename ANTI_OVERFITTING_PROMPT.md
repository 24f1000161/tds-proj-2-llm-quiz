# PROJECT 2: ANTI-OVERFITTING PROMPT FOR LLM AGENT

**YOU MUST READ AND FOLLOW THIS COMPLETELY.**

**Current Problem**: Your `pipeline.py` is OVERFITTED:
- ❌ Hardcoded question patterns (`"email offset"`, `"cutoff"`, etc.)
- ❌ Static answer detection with regex (not dynamic)
- ❌ Hardcoded GitHub, git, UV, ZIP logic
- ❌ Does NOT use LLM to detect answer format
- ❌ Does NOT use LLM to decide WHICH FUNCTION to run
- ❌ Only works for known question types from Q834, Q942, etc.

**Solution**: REBUILD with dynamic LLM-driven logic

---

## CRITICAL REPO STRUCTURE TO UNDERSTAND

Read this ENTIRE repository first:
**https://github.com/sanand0/tools-in-data-science-public/**

This course has:
- **7 Modules**: Development Tools, Deployment, LLMs, Data Sourcing, Data Preparation, Analysis, Visualization
- **Question Types**: Data analysis, web scraping, LLM tasks, tool usage, git/shell commands, API calls, audio transcription, multi-modal
- **NOT predictable**: Each question is UNIQUE. Formats vary wildly.

---

## REQUIRED CHANGES TO PIPELINE.PY

### 1. ❌ REMOVE ALL HARDCODED PATTERNS

Delete these functions that memorize specific question types:
```python
# DELETE THIS:
def extract_answer_from_context(merged_context, question_text):
    """Pattern matching for 'secret code', numbers in <strong>, etc."""
    # ❌ REMOVE - This hardcodes specific quiz patterns
    
# DELETE THIS:
if merged_context.get('dominant_color') and ('color' in raw_question.lower() or 'hex' in raw_question.lower()):
    answer = merged_context['dominant_color']
    
# DELETE THIS:
elif merged_context.get('github_config') and ('count' in raw_question.lower() or '.md' in raw_question.lower()):
    # ❌ Hardcoded GitHub logic
    
# DELETE THIS:
elif 'uv' in raw_question.lower() and 'http' in raw_question.lower():
    # ❌ Hardcoded UV command logic
    
# DELETE THIS:
elif 'git' in raw_question.lower() and 'commit' in raw_question.lower():
    # ❌ Hardcoded git command logic
```

### 2. ✅ REPLACE WITH LLM-DRIVEN LOGIC

All decision-making MUST go through LLM:

```python
async def solve_quiz_pipeline(...):
    # ... existing setup ...
    
    # NEW: Use LLM to understand the question COMPLETELY
    question_analysis = await llm_client.analyze_question_deeply(raw_question)
    # Returns:
    # {
    #   "task_type": "data_analysis" | "web_scrape" | "command" | "llm_task" | "api" | "other",
    #   "answer_format": "number" | "string" | "json" | "command" | "code" | "image" | "boolean",
    #   "required_data": ["url1", "url2"],
    #   "data_sources_implicit": ["Wikipedia", "API endpoint", "file upload"],
    #   "solution_strategy": "Use pandas to compute mean()",
    #   "confidence": 0.95,
    #   "edge_cases": ["handle NaN", "round to 2 decimals"],
    # }
    
    logger.info(f"Question Analysis: {question_analysis}")
    
    # NEW: Based on analysis, select the RIGHT FUNCTION
    answer = None
    
    if question_analysis['task_type'] == 'data_analysis':
        answer = await llm_driven_data_analysis(
            df, merged_context, raw_question, question_analysis
        )
    elif question_analysis['task_type'] == 'web_scrape':
        answer = await llm_driven_web_scrape(
            page, raw_question, question_analysis
        )
    elif question_analysis['task_type'] == 'command':
        answer = await llm_driven_command_generation(
            raw_question, question_analysis
        )
    elif question_analysis['task_type'] == 'llm_task':
        answer = await llm_driven_llm_task(
            merged_context, raw_question, question_analysis
        )
    elif question_analysis['task_type'] == 'api':
        answer = await llm_driven_api_call(
            raw_question, question_analysis
        )
    else:
        # Fallback: Use LLM directly
        answer = await llm_client.answer_from_context(
            merged_context, raw_question
        )
    
    # NEW: Use LLM to detect answer format
    formatted_answer = await llm_client.format_answer_dynamically(
        answer, question_analysis['answer_format'], raw_question
    )
```

### 3. ✅ IMPLEMENT DYNAMIC ANALYSIS FUNCTIONS

**llm_driven_data_analysis():**
```python
async def llm_driven_data_analysis(df, context, question, analysis):
    """
    Uses LLM to DECIDE what to do, not hardcoded logic.
    """
    
    # Ask LLM: "What columns? What operation? How to validate?"
    solution_code = await llm_client.generate_analysis_code(
        dataframe_info=get_dataframe_info(df),
        question=question,
        context=context,
        guidance=analysis.get('solution_strategy'),
        edge_cases=analysis.get('edge_cases')
    )
    
    # Execute generated code
    answer, output, error = await execute_analysis_code(solution_code, df)
    
    if error:
        logger.warning(f"First attempt failed: {error}")
        # Ask LLM to fix the code
        fixed_code = await llm_client.fix_analysis_code(
            original_code=solution_code,
            error=error,
            dataframe_info=get_dataframe_info(df),
            question=question
        )
        answer, output, error = await execute_analysis_code(fixed_code, df)
    
    return answer
```

**llm_driven_web_scrape():**
```python
async def llm_driven_web_scrape(page, question, analysis):
    """
    Uses LLM to decide:
    - What to scrape?
    - How to extract?
    - What to return?
    """
    
    # Ask LLM what to scrape
    scrape_plan = await llm_client.plan_scraping(
        page_content=await page.evaluate("document.body.innerText"),
        question=question
    )
    # Returns: {"targets": [...], "selectors": [...], "extraction_method": "..."}
    
    # Execute scraping based on plan
    scraped_data = await execute_scrape_plan(page, scrape_plan)
    
    # Ask LLM to extract answer from scraped data
    answer = await llm_client.extract_answer_from_scraped(
        scraped_data, question
    )
    
    return answer
```

**llm_driven_command_generation():**
```python
async def llm_driven_command_generation(question, analysis):
    """
    Don't hardcode git/UV/shell commands.
    Let LLM generate them from question.
    """
    
    command = await llm_client.generate_command(
        question=question,
        command_type=analysis.get('solution_strategy'),
        validate=True  # Ensure command is valid
    )
    
    return command
```

**llm_driven_api_call():**
```python
async def llm_driven_api_call(question, analysis):
    """
    Uses LLM to:
    - Find API endpoint
    - Generate correct request
    - Parse response
    """
    
    api_plan = await llm_client.plan_api_call(
        question=question,
        guidance=analysis.get('solution_strategy')
    )
    # Returns: {"method": "GET", "url": "...", "headers": {...}, "params": {...}}
    
    response = await make_api_call(api_plan)
    
    answer = await llm_client.extract_answer_from_api(
        response, question
    )
    
    return answer
```

### 4. ✅ LLM MUST DETECT ANSWER FORMAT

**NEW in llm_client:**
```python
async def detect_answer_format_dynamically(answer, question):
    """
    Don't assume answer is a number or string.
    Ask LLM what format it should be.
    """
    
    format_detection = await self.call_aipipe(
        system="""You are an expert at detecting required answer formats.
        Given a question and a computed answer, return the format it should be in.
        
        Formats: number, string, boolean, json, command, code, image, csv, list
        
        Return JSON: {"format": "...", "precision": ..., "validation": "..."}
        """,
        messages=[
            {"role": "user", "content": f"""
            Question: {question}
            Computed answer: {answer}
            
            What format should this be submitted in?
            """}
        ]
    )
    
    return format_detection  # Dynamically format based on this
```

---

## TESTING: PROVE NO OVERFITTING

Test with DIVERSE questions from the course repo:

```python
test_questions = [
    # Type 1: Simple stats
    "Download CSV, compute mean of column X",
    
    # Type 2: PDF extraction
    "Extract table from PDF page 2",
    
    # Type 3: Web scrape
    "Scrape Wikipedia for Indian population",
    
    # Type 4: LLM task
    "Use LLM to classify these items",
    
    # Type 5: Git command
    "Stage file X with commit message Y",
    
    # Type 6: Audio
    "Transcribe audio file and answer the question",
    
    # Type 7: Image color
    "What is the dominant color in this image?",
    
    # Type 8: API call
    "Call GitHub API to count .md files",
    
    # Type 9: JSON transformation
    "Normalize CSV to JSON with snake_case columns",
    
    # Type 10: Multi-step
    "Download data, clean it, analyze it, return JSON",
]

for q in test_questions:
    # Run solver on each
    answer = await solve(q)
    assert answer is not None
    print(f"✅ {q[:50]}... → {answer}")
```

**If ANY fail, you're OVERFITTING.**

---

## CODE STRUCTURE RECOMMENDATION

```
pipeline.py (MINIMAL)
├── solve_quiz_pipeline()  [Main orchestration]
└── Delegates to LLM for ALL decisions

llm_analysis.py (NEW - LLM-DRIVEN LOGIC)
├── analyze_question_deeply()
├── llm_driven_data_analysis()
├── llm_driven_web_scrape()
├── llm_driven_command_generation()
├── llm_driven_api_call()
├── llm_driven_llm_task()
└── detect_answer_format_dynamically()

analysis.py (EXECUTION ONLY)
├── execute_analysis_code()  [Just runs code]
├── execute_scrape_plan()    [Just scrapes]
└── make_api_call()          [Just calls API]
```

**Key principle**: 
- `pipeline.py` = Orchestration (calls LLM to decide)
- `llm_analysis.py` = LLM decision-making (NOT hardcoded)
- `analysis.py` = Execution (runs what LLM decides)

---

## CHECKLIST: BEFORE SUBMISSION

- [ ] Zero hardcoded question patterns (no "if 'email' in question")
- [ ] LLM decides task type for EVERY question
- [ ] LLM generates solution code (not hardcoded)
- [ ] LLM detects answer format dynamically
- [ ] All functions are generic (work for unknown questions)
- [ ] Pipeline uses decision tree from LLM, not static logic
- [ ] Tested on diverse question types (10+ different formats)
- [ ] No regex-based answer extraction
- [ ] Error recovery uses LLM (not fallback hardcoding)
- [ ] Every stage logs decision rationale

---

## KEY MINDSET SHIFT

**WRONG:**
```python
if 'github' in question and 'count' in question:
    # Call GitHub API with hardcoded logic
```

**RIGHT:**
```python
# LLM understands question, decides it needs GitHub API
# LLM generates the exact API call needed
# LLM tells us to count and add email offset
# ALL via LLM, zero hardcoding
```

**The agent should be CAPABLE of solving ANY question in the course.**
**NOT memorizing specific questions.**

---

## REFERENCES

- Course repo: https://github.com/sanand0/tools-in-data-science-public/
- 7 modules, ANY question type possible
- Your job: Build a generalizable solver
- NOT a pattern matcher

Start rebuilding immediately. This is the core issue.
