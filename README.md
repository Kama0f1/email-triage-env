# Email Triage OpenEnv

An RL environment where an AI agent learns to triage emails — labelling, prioritizing, and drafting replies.

## Why Email Triage?

Email overload is a universal problem. This environment trains agents to handle it intelligently, with real-world applicability for productivity tools and enterprise AI assistants.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| task1 | Easy | Label a single email: spam / work / personal / urgent |
| task2 | Medium | Rank 5 emails by urgency |
| task3 | Hard | Label + prioritize + draft a professional reply |

## Action Space

```json
{
  "label": "spam | work | personal | urgent",
  "ranking": ["e1", "e3", "e5", "e2", "e4"],
  "reply": "Dear sender, ..."
}
```

## Observation Space

```json
{
  "task_id": "task1",
  "task_description": "Label the email...",
  "email_subject": "...",
  "email_sender": "...",
  "email_body": "...",
  "emails": [],
  "feedback": "Correct!",
  "reward": 1.0,
  "done": true
}
```

## Reward Function

- **Task 1**: 1.0 for correct label, 0.0 otherwise
- **Task 2**: Partial credit per correct position (0.0–1.0) + 0.1 bonus if top 2 are correct
- **Task 3**: Weighted score — label (30%) + ranking (30%) + reply keyword coverage (40%)

## Baseline Scores (gpt-4o-mini)

| Task | Score |
|------|-------|
| task1 (easy) | 1.00 |
| task2 (medium) | 0.72 |
| task3 (hard) | 0.58 |
| **Average** | **0.77** |

## Setup & Usage

### Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key email-triage-env
```

### Run baseline

```bash
OPENAI_API_KEY=your_key python baseline.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks + action schema |
| `/grader` | POST | Score an action |
| `/baseline` | GET | Run baseline inference |
