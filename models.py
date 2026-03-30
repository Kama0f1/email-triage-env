from pydantic import BaseModel, Field
from typing import Optional, List


class EmailAction(BaseModel):
    """What the agent can do at each step."""
    action_type: str = ""
    label: Optional[str] = None
    ranking: Optional[List[str]] = Field(default_factory=list)
    reply: Optional[str] = None


class EmailObservation(BaseModel):
    """What the agent sees at each step."""
    task_id: str = ""
    task_description: str = ""
    email_subject: Optional[str] = None
    email_sender: Optional[str] = None
    email_body: Optional[str] = None
    emails: Optional[List[dict]] = Field(default_factory=list)
    feedback: Optional[str] = None
    reward: float = 0.0
    done: bool = False
    valid_labels: List[str] = Field(
        default_factory=lambda: ["spam", "work", "personal", "urgent"]
    )


class EmailState(BaseModel):
    """Episode metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    current_score: float = 0.0
    max_steps: int = 5