"""
FastAPI server exposing the Email Triage environment.
Required endpoints: /reset  /step  /state  /tasks  /grader  /baseline
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import subprocess, sys, os

from environment import EmailTriageEnvironment
from models import EmailAction
from tasks import TASKS
from graders import grade

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An RL environment for email triage tasks.",
    version="1.0.0",
)

# Single shared environment instance
env = EmailTriageEnvironment()


# ── Request/Response schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1"

class StepRequest(BaseModel):
    action_type: str = ""
    label: Optional[str] = None
    ranking: Optional[List[str]] = []
    reply: Optional[str] = None

class GraderRequest(BaseModel):
    task_id: str
    label: Optional[str] = None
    ranking: Optional[List[str]] = []
    reply: Optional[str] = None


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest):
    obs = env.reset(task_id=req.task_id)
    return obs.__dict__


@app.post("/step")
def step(req: StepRequest):
    action = EmailAction(
        action_type=req.action_type,
        label=req.label,
        ranking=req.ranking or [],
        reply=req.reply,
    )
    obs = env.step(action)
    state = env.state()
    return {
        "observation": obs.__dict__,
        "reward": obs.reward,
        "done": obs.done,
        "info": {"step_count": state.step_count},
    }


@app.get("/state")
def state():
    return env.state().__dict__


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
def grader(req: GraderRequest):
    """Score an action against the correct answer for a task."""
    if req.task_id not in TASKS:
        return JSONResponse(status_code=400, content={"error": "Invalid task_id"})

    action_dict = {
        "label": req.label,
        "ranking": req.ranking or [],
        "reply": req.reply,
    }
    score = grade(req.task_id, action_dict, TASKS[req.task_id])
    return {
        "task_id": req.task_id,
        "score": score,
        "max_score": 1.0,
    }


@app.get("/baseline")
def baseline():
    """
    Trigger the inference script.
    Runs the model against all 3 tasks and returns scores.
    Requires: API_BASE_URL, MODEL_NAME, HF_TOKEN env variables.
    """
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
            timeout=1200,  # 20 min max per requirements
            env={
                **os.environ,
                "HF_TOKEN": hf_token,
                "API_BASE_URL": os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
                "MODEL_NAME": os.environ.get("MODEL_NAME", "gpt-4o-mini"),
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