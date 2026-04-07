"""
Inference script for Email Triage OpenEnv.
Uses OpenAI client pointed at HF router (free).

Required environment variables:
  HF_TOKEN      - Your Hugging Face token
  API_BASE_URL  - LLM API base URL (default: https://router.huggingface.co/v1)
  MODEL_NAME    - Model to use (default: mistralai/Mistral-7B-Instruct-v0.3)
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

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)


def build_prompt(task: dict) -> str:
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
    return {}


def run_task(task_id: str, task: dict) -> float:
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
        print(f"  [!] Model call failed: {e}", flush=True)
        return 0.0

    action = extract_json(raw)
    if not action:
        return 0.0
    return grade(task_id, action, task)


def main():
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(f"Base URL: {API_BASE_URL}", flush=True)

    for task_id, task in TASKS.items():
        # Print START block
        print(f"[START] task={task_id}", flush=True)

        score = 0.0
        step = 0

        try:
            score = run_task(task_id, task)
            step = 1
            # Print STEP block
            print(f"[STEP] step={step} reward={score:.2f}", flush=True)
        except Exception as e:
            print(f"[STEP] step=1 reward=0.00", flush=True)
            print(f"Error: {e}", flush=True)

        # Print END block
        print(f"[END] task={task_id} score={score:.2f} steps={step}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
