"""
Inference script for Email Triage OpenEnv.
Must be named inference.py per submission requirements.

Required environment variables:
  API_BASE_URL  - The API endpoint for the LLM
  MODEL_NAME    - The model identifier to use
  HF_TOKEN      - Your Hugging Face / API key

Usage:
  API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py
"""

import os
import json
from openai import OpenAI
from tasks import TASKS
from graders import grade

# ── Read required env variables ───────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "Please set it to your OpenAI or HuggingFace API key."
    )

# ── OpenAI client using the required variables ────────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)


def build_prompt(task: dict) -> str:
    """Build a prompt describing the task for the agent."""
    lines = [
        f"TASK: {task['description']}",
        "",
    ]

    if task["email"]:
        e = task["email"]
        lines += [
            "EMAIL:",
            f"  From: {e['sender']}",
            f"  Subject: {e['subject']}",
            f"  Body: {e['body']}",
            "",
        ]

    if task.get("emails"):
        lines.append("EMAILS IN INBOX:")
        for e in task["emails"]:
            lines.append(
                f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}"
            )
        lines.append("")

    lines += [
        "Respond ONLY with a JSON object with these fields (use null if not needed):",
        '{',
        '  "label": "spam" | "work" | "personal" | "urgent" | null,',
        '  "ranking": ["e1", "e2", ...] | null,',
        '  "reply": "your drafted reply" | null',
        '}',
    ]
    return "\n".join(lines)


def run_task(task_id: str, task: dict) -> float:
    """Run the model on one task and return the score."""
    prompt = build_prompt(task)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert email triage assistant. "
                    "Respond only with valid JSON, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [!] Failed to parse JSON for {task_id}: {raw[:100]}")
        return 0.0

    score = grade(task_id, action, task)
    return score


def main():
    print("=" * 50)
    print("Email Triage — Inference Script")
    print(f"Model:    {MODEL_NAME}")
    print(f"Base URL: {API_BASE_URL}")
    print("=" * 50)

    results = {}
    for task_id, task in TASKS.items():
        print(f"\nRunning {task_id} ({task['difficulty']})...")
        score = run_task(task_id, task)
        results[task_id] = score
        print(f"  Score: {score:.2f} / 1.00")

    print("\n" + "=" * 50)
    print("RESULTS:")
    for task_id, score in results.items():
        difficulty = TASKS[task_id]["difficulty"]
        print(f"  {task_id} ({difficulty}): {score:.2f}")
    avg = sum(results.values()) / len(results)
    print(f"  Average: {avg:.2f}")
    print("=" * 50)

    # Return results dict so /baseline endpoint can call this
    return results


if __name__ == "__main__":
    main()
