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

## API Endpoints

- `POST /api/quiz` - Submit a quiz to solve
- `GET /health` - Health check
- `GET /logs` - View recent logs
- `DELETE /logs` - Clear logs

## Configuration

Set the following secrets in your Hugging Face Space settings:

**Required:**
- `STUDENT_SECRETS` - Email:secret pairs (comma-separated), e.g. `student@example.com:mysecret`
- `GEMINI_API_KEY` - Google Gemini API key (get from https://aistudio.google.com/app/apikey)

**Optional (for fallback):**
- `AIPIPE_TOKEN` - AI Pipe token for OpenAI fallback (get from https://aipipe.org)
- `AIPIPE_MODEL` - Model to use via aipipe (default: `openai/gpt-4o-mini`)

**Model Configuration:**
- `GEMINI_MODEL` - Gemini model to use (default: `gemini-2.0-flash`)
- `GEMINI_VIA_AIPIPE` - Set to `false` to use direct Gemini API (recommended)

## Usage

```bash
curl -X POST https://YOUR-SPACE.hf.space/api/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your-secret",
    "url": "https://example.com/quiz"
  }'
```
