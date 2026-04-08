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
from typing import List, Optional
from openai import OpenAI
from tasks import TASKS
from graders import grade, clamp

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
BENCHMARK    = os.environ.get("BENCHMARK", "email-triage-env")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_prompt(task: dict) -> str:
    lines = [f"TASK: {task['description']}", ""]
    if task["email"]:
        e = task["email"]
        lines += ["EMAIL:", f"  From: {e['sender']}", f"  Subject: {e['subject']}", f"  Body: {e['body']}", ""]
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


def main():
    for task_id, task in TASKS.items():
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        steps_taken = 0
        score = 0.05
        success = False
        error_msg = None
        action_str = "null"

        try:
            prompt = build_prompt(task)
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert email triage assistant. Respond only with valid JSON, no extra text."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )
                raw = response.choices[0].message.content.strip()
            except Exception as e:
                raw = ""
                error_msg = str(e)

            action = extract_json(raw)
            action_str = json.dumps(action) if action else "null"

            raw_score = grade(task_id, action, task) if action else 0.05
            step_reward = clamp(raw_score)  # always strictly (0.01, 0.99)

            rewards.append(step_reward)
            steps_taken = 1
            score = step_reward
            success = score >= 0.5

            log_step(step=1, action=action_str, reward=step_reward, done=True, error=error_msg)

        except Exception as e:
            step_reward = 0.05
            rewards.append(step_reward)
            steps_taken = 1
            score = 0.05
            log_step(step=1, action="null", reward=step_reward, done=True, error=str(e))

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
