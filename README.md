---
title: Email Triage Env
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Email Triage OpenEnv

An RL environment where an AI agent learns to triage emails — labelling, prioritizing, and drafting replies. Built for the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Why Email Triage?

Email overload is one of the most universal productivity problems. Every professional deals with it daily. This environment trains AI agents to handle it intelligently — making it immediately useful for evaluating LLM-based assistants, productivity tools, and enterprise AI systems.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `task1` | Easy | Label a single email as `spam`, `work`, `personal`, or `urgent` |
| `task2` | Medium | Rank 5 emails from most to least urgent |
| `task3` | Hard | Label the email + rank all 3 inbox emails + draft a professional reply |

## Action Space

```json
{
  "action_type": "string (optional)",
  "label": "spam | work | personal | urgent | null",
  "ranking": ["e1", "e3", "e5", "e2", "e4"],
  "reply": "Dear sender, thank you for reaching out..."
}
```

## Observation Space

```json
{
  "task_id": "task1",
  "task_description": "Label the email correctly...",
  "email_subject": "CONGRATULATIONS! You won $1,000,000!!!",
  "email_sender": "winner@totallylegit-prizes.com",
  "email_body": "Dear Lucky Winner...",
  "emails": [],
  "feedback": "Perfect! Full marks.",
  "reward": 1.0,
  "done": true,
  "valid_labels": ["spam", "work", "personal", "urgent"]
}
```

## Reward Function

| Task | Scoring Logic |
|------|--------------|
| Task 1 | 1.0 for correct label, 0.0 otherwise |
| Task 2 | Partial credit per correct position (0.0-1.0) + 0.1 bonus if top 2 are correct |
| Task 3 | Weighted: label (30%) + ranking (30%) + reply keyword coverage (40%) |

Rewards are dense — partial progress is always rewarded, never just binary win/lose.

## Baseline Scores

Evaluated using mistralai/Mistral-7B-Instruct-v0.3 via HF Router:

| Task | Difficulty | Score |
|------|-----------|-------|
| task1 | Easy | 1.00 |
| task2 | Medium | 0.72 |
| task3 | Hard | 0.58 |
| Average | | 0.77 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode for a given task |
| `/step` | POST | Agent takes an action, returns observation + reward |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Score an action against correct answer |
| `/baseline` | GET | Trigger inference script, returns scores |
| `/docs` | GET | Interactive API docs (Swagger UI) |

## Setup & Usage

### Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Visit http://localhost:7860/docs to test interactively.

### Run with Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_hf_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  email-triage-env
```

### Run inference script

```bash
HF_TOKEN=your_hf_token \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
python inference.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HF_TOKEN | Hugging Face API token | required |
| API_BASE_URL | LLM API base URL | https://router.huggingface.co/v1 |
| MODEL_NAME | Model identifier | mistralai/Mistral-7B-Instruct-v0.3 |

## Project Structure

```
email-triage-env/
├── app.py            # FastAPI server with all endpoints
├── environment.py    # Core reset/step/state logic
├── models.py         # Pydantic action/observation/state models
├── graders.py        # Deterministic graders scoring 0.0-1.0
├── tasks.py          # 3 task definitions with sample emails
├── inference.py      # Baseline inference script
├── openenv.yaml      # OpenEnv manifest
├── requirements.txt  # Dependencies
├── Dockerfile        # Container definition
└── README.md         # This file
```

## Live Demo

Try it on Hugging Face Spaces: https://huggingface.co/spaces/Kama0f1/email-triage-env

Interactive docs: https://kama0f1-email-triage-env.hf.space/docs