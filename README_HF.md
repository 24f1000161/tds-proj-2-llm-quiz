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

- `STUDENT_SECRETS` - Email:secret pairs (comma-separated)
- `AIPIPE_TOKEN` - AI Pipe token for LLM access

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
