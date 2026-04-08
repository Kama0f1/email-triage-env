"""
FastAPI server exposing the Email Triage environment.
All reward/score values are strictly between 0 and 1 (never exactly 0.0 or 1.0).
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import subprocess, sys, os

from environment import EmailTriageEnvironment
from models import EmailAction
from tasks import TASKS
from graders import grade, clamp

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An RL environment for email triage tasks.",
    version="1.0.0",
)

env = EmailTriageEnvironment()


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: Request):
    """Accept empty body or JSON body with optional task_id."""
    try:
        body = await request.json()
        task_id = body.get("task_id", "task1") if body else "task1"
    except Exception:
        task_id = "task1"
    obs = env.reset(task_id=task_id)
    result = obs.model_dump()
    result["reward"] = clamp(result.get("reward", 0.05))
    return result


@app.post("/step")
async def step(request: Request):
    """Accept action and return observation + reward."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    action = EmailAction(
        action_type=body.get("action_type", ""),
        label=body.get("label", None),
        ranking=body.get("ranking", []),
        reply=body.get("reply", None),
    )
    obs = env.step(action)
    state = env.state()
    reward = clamp(obs.reward)
    obs_dict = obs.model_dump()
    obs_dict["reward"] = reward
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": obs.done,
        "info": {"step_count": state.step_count},
    }


@app.get("/state")
def state():
    s = env.state().model_dump()
    s["current_score"] = clamp(s.get("current_score", 0.05))
    return s


# ── Required extra endpoints ──────────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    """Returns list of tasks and the action schema."""
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": t["difficulty"],
                "description": t["description"],
            }
            for tid, t in TASKS.items()
        ],
        "action_schema": {
            "action_type": "string (optional)",
            "label": "string — one of: spam, work, personal, urgent",
            "ranking": "list of email IDs in priority order e.g. ['e1','e3','e2']",
            "reply": "string — drafted email reply",
        },
    }


@app.post("/grader")
async def grader(request: Request):
    """Score an action against the correct answer for a task."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = body.get("task_id", "task1")
    if task_id not in TASKS:
        task_id = "task1"
    action_dict = {
        "label": body.get("label"),
        "ranking": body.get("ranking", []),
        "reply": body.get("reply"),
    }
    score = clamp(grade(task_id, action_dict, TASKS[task_id]))
    return {
        "task_id": task_id,
        "score": score,
        "max_score": 0.99,
    }


@app.get("/baseline")
def baseline():
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return JSONResponse(
            status_code=400,
            content={"error": "HF_TOKEN not set in environment"},
        )
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=1200,
            env={
                **os.environ,
                "HF_TOKEN": hf_token,
                "API_BASE_URL": os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
                "MODEL_NAME": os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
            },
        )
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"error": "Inference timed out"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }
