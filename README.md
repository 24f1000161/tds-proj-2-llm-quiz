---
title: LLM Quiz Solver
emoji: "🤖"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# LLM Quiz Solver

An automated quiz solving agent powered by LLM with 3-minute deadline support.

## Features

- **Dynamic Task Classification**: LLM analyzes questions to determine task type, answer format, and personalization needs
- **Generic Task Handlers**: 9 specialized handlers that work for ANY question variation (no hardcoded patterns)
- **FastAPI-based REST API**: Production-ready endpoint for quiz submissions
- **Playwright browser automation**: Full support for Vue.js/React/Angular SPAs with dynamic content
- **Dual LLM strategy**: aipipe + Gemini fallback via [aipipe.org](https://aipipe.org)
- **Multi-modal support**: Audio transcription (Gemini), image analysis, PDF extraction
- **Automatic data extraction**: PDFs, CSVs, Excel, JSON, ZIP files
- **Smart personalization**: Automatic detection and calculation of email-based offsets
- **3-minute deadline enforcement**: Strict timing with safety buffers
- **Chained quiz support**: Follows quiz chains until completion

## Project Structure

```
quiz_solver/
├── __init__.py          # Package initialization
├── api.py               # FastAPI application
├── main.py              # Entry point
├── config.py            # Configuration management
├── models.py            # Pydantic models
├── pipeline.py          # Main quiz solving pipeline (9-stage)
├── browser.py           # Playwright browser automation
├── llm_client.py        # LLM client with dynamic task classification
├── llm_analysis.py      # LLM-driven analysis with dynamic routing
├── task_handlers.py     # Generic handlers for all task types (NEW)
├── data_sourcing.py     # Data fetching and parsing
├── question_parser.py   # Question extraction utilities
├── analysis.py          # Data analysis execution
├── submission.py        # Answer submission
└── logging_utils.py     # Logging utilities
```

## Architecture

### Dynamic Task Routing (NEW)

The system uses a 3-step intelligent routing approach:

1. **LLM Classification**: Analyzes question to determine:
   - Task type (image_analysis, api_call, data_analysis, command_generation, etc.)
   - Answer format (hex_color, integer, json, command_string, etc.)
   - Personalization requirements (email_length_mod_2, etc.)

2. **Handler Routing**: Routes to appropriate generic handler:
   - `handle_image_task()` - ANY image question
   - `handle_api_task()` - ANY API call (GitHub, REST, custom)
   - `handle_data_analysis_task()` - ANY pandas operation
   - `handle_command_task()` - ANY shell command
   - And 5 more specialized handlers

3. **Generic Execution**: Handler solves question using LLM guidance
   - Works for ANY phrasing or variation
   - No hardcoded keyword matching
   - Automatic personalization offset calculation

## Installation

This project uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
# Install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Configure your environment variables:
```env
# Student secrets (email:secret pairs)
STUDENT_SECRETS=student@example.com:mysecret

# AI Pipe Token (get from https://aipipe.org/login)
AIPIPE_TOKEN=your-aipipe-token

# Optional: Direct Gemini API key (if not using aipipe proxy)
GEMINI_API_KEY=your-gemini-key
GEMINI_VIA_AIPIPE=false
```

## Running

### Development
```bash
uv run python -m quiz_solver.main
```

Or using the script entry point:
```bash
uv run quiz-solver
```

### Production
```bash
uv run uvicorn quiz_solver.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /api/quiz

Submit a quiz to be solved.

**Request:**
```json
{
  "email": "student@example.com",
  "secret": "your-secret",
  "url": "https://example.com/quiz-123"
}
```

**Response (200 OK):**
```json
{
  "status": "accepted",
  "quiz_id": "abc123def456",
  "timestamp": "2025-11-28T10:00:00.000000"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid JSON payload
- `403 Forbidden`: Invalid email or secret

### GET /health

Health check endpoint.

## LLM Configuration

This project uses [AI Pipe](https://aipipe.org) for LLM access:

- **Primary**: GPT-4o-mini via aipipe OpenAI-compatible API
- **Fallback**: Gemini 2.0 Flash via aipipe OpenRouter proxy

Get your AI Pipe token from [aipipe.org/login](https://aipipe.org/login).

## Testing

```bash
# Test with curl
curl -X POST http://localhost:8000/api/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "student@example.com",
    "secret": "your-secret",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

## License

MIT
