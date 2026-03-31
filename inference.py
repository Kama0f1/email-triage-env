"""
Inference script for Email Triage OpenEnv.
Uses OpenAI client pointed at HF router (free).

Required environment variables:
  HF_TOKEN      - Your Hugging Face token (free, starts with hf_)
  API_BASE_URL  - LLM API base URL (default: https://router.huggingface.co/v1)
  MODEL_NAME    - Model to use (default: mistralai/Mistral-7B-Instruct-v0.3)

Usage:
  HF_TOKEN=hf_xxx python inference.py
"""

import os
import json
from openai import OpenAI
from tasks import TASKS
from graders import grade

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

# OpenAI client pointed at HF router — free, no credit card needed
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)


def build_prompt(task: dict) -> str:
    """Build a prompt for the agent."""
    lines = [f"TASK: {task['description']}", ""]
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
            lines.append(f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}")
        lines.append("")
    lines += [
        "Respond ONLY with a JSON object. No explanation. No markdown. Just JSON:",
        '{"label": "spam or work or personal or urgent or null", "ranking": ["e1","e2",...] or null, "reply": "text or null"}',
    ]
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Extract JSON from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    print(f"  [!] Could not extract JSON from: {text[:150]}")
    return {}


def run_task(task_id: str, task: dict) -> float:
    """Run the model on one task and return the score."""
    prompt = build_prompt(task)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert email triage assistant. Respond only with valid JSON, no extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [!] Model call failed: {e}")
        return 0.0

    action = extract_json(raw)
    if not action:
        return 0.0
    return grade(task_id, action, task)


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
    return results


if __name__ == "__main__":
    main()