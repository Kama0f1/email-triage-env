"""
The Email Triage Environment.
Implements reset(), step(), state() as required by OpenEnv spec.
"""

import uuid
from models import EmailAction, EmailObservation, EmailState
from tasks import TASKS
from graders import grade


class EmailTriageEnvironment:

    def __init__(self):
        self._state = EmailState()
        self._current_task = None
        self._last_action = {}

    def reset(self, task_id: str = "task1") -> EmailObservation:
        """Start a fresh episode for the given task."""
        if task_id not in TASKS:
            task_id = "task1"

        self._current_task = TASKS[task_id]
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            current_score=0.0,
        )
        self._last_action = {}

        task = self._current_task
        return EmailObservation(
            task_id=task_id,
            task_description=task["description"],
            email_subject=task["email"]["subject"] if task["email"] else None,
            email_sender=task["email"]["sender"] if task["email"] else None,
            email_body=task["email"]["body"] if task["email"] else None,
            emails=task.get("emails") or [],
            feedback="Episode started. Read the email(s) and take action.",
            reward=0.0,
            done=False,
        )

    def step(self, action: EmailAction) -> EmailObservation:
        """Agent takes an action. Returns observation + reward."""
        if self._current_task is None:
            # Auto-reset to task1 if not initialized
            return self.reset("task1")

        self._state.step_count += 1
        task = self._current_task
        task_id = self._state.task_id

        action_dict = {
            "label": action.label,
            "ranking": action.ranking,
            "reply": action.reply,
        }

        score = grade(task_id, action_dict, task)
        self._state.current_score = score
        self._last_action = action_dict

        # Build feedback message
        if score == 1.0:
            feedback = "Perfect! Full marks."
        elif score >= 0.7:
            feedback = f"Good job! Score: {score}. Some parts could be improved."
        elif score >= 0.4:
            feedback = f"Partial credit. Score: {score}. Review your answer."
        else:
            feedback = f"Incorrect. Score: {score}. Try again."

        # Episode ends after 1 step (each task is single-turn)
        done = True

        return EmailObservation(
            task_id=task_id,
            task_description=task["description"],
            email_subject=task["email"]["subject"] if task["email"] else None,
            email_sender=task["email"]["sender"] if task["email"] else None,
            email_body=task["email"]["body"] if task["email"] else None,
            emails=task.get("emails") or [],
            feedback=feedback,
            reward=score,
            done=done,
        )

    def state(self) -> EmailState:
        """Return current episode state."""
        return self._state