# LLM Quiz Solver

An automated quiz solving agent powered by LLM with 3-minute deadline support.

## Features

- FastAPI-based REST API
- Playwright browser automation for Vue.js/React/Angular SPAs
- Dual LLM strategy (aipipe + Gemini fallback via [aipipe.org](https://aipipe.org))
- Audio transcription support (via Gemini)
- Automatic data extraction from PDFs, CSVs, Excel files
- Dynamic LLM-generated analysis code
- 3-minute per-question deadline enforcement
- Chained quiz support

## Project Structure

```
quiz_solver/
├── __init__.py          # Package initialization
├── api.py               # FastAPI application
├── main.py              # Entry point
├── config.py            # Configuration management
├── models.py            # Pydantic models
├── pipeline.py          # Main quiz solving pipeline
├── browser.py           # Playwright browser automation
├── llm_client.py        # LLM client with dual-model strategy
├── data_sourcing.py     # Data fetching and parsing
├── question_parser.py   # Question extraction utilities
├── analysis.py          # Data analysis execution
├── submission.py        # Answer submission
└── logging_utils.py     # Logging utilities
```

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
