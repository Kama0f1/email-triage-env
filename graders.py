"""
Graders for each task. Each returns a float strictly between 0.0 and 1.0.
Deterministic and reproducible — same inputs always give same score.
"""

from typing import List, Optional


def clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (never exactly 0.0 or 1.0)."""
    return round(max(0.01, min(0.99, score)), 4)


def grade_task1(label: Optional[str], correct_label: str) -> float:
    """
    Task 1 (easy): Label the email.
    - Correct label → 0.99
    - Wrong label   → 0.01
    """
    if label is None:
        return 0.01
    return 0.99 if label.strip().lower() == correct_label.lower() else 0.01


def grade_task2(ranking: Optional[List[str]], correct_ranking: List[str]) -> float:
    """
    Task 2 (medium): Prioritize 5 emails.
    Partial credit based on how many emails are in the correct position.
    """
    if not ranking:
        return 0.01

    n = len(correct_ranking)
    if len(ranking) != n:
        correct = sum(
            1 for i, eid in enumerate(ranking)
            if i < n and eid == correct_ranking[i]
        )
        return clamp(correct / n)

    correct = sum(1 for i in range(n) if ranking[i] == correct_ranking[i])

    top2_correct = ranking[:2] == correct_ranking[:2]
    bonus = 0.1 if top2_correct else 0.0

    score = (correct / n) + bonus
    return clamp(score)


def grade_task3(
    label: Optional[str],
    ranking: Optional[List[str]],
    reply: Optional[str],
    correct_label: str,
    correct_ranking: List[str],
    correct_reply_keywords: List[str],
) -> float:
    """
    Task 3 (hard): Label + prioritize + reply.
    Weighted: label (30%) + ranking (30%) + reply (40%)
    """
    label_score = 0.99 if (label and label.strip().lower() == correct_label.lower()) else 0.01
    ranking_score = grade_task2(ranking, correct_ranking)

    if not reply:
        reply_score = 0.01
    else:
        reply_lower = reply.lower()
        matched = sum(1 for kw in correct_reply_keywords if kw in reply_lower)
        reply_score = clamp(matched / len(correct_reply_keywords))

    total = (label_score * 0.30) + (ranking_score * 0.30) + (reply_score * 0.40)
    return clamp(total)


def grade(task_id: str, action: dict, task_data: dict) -> float:
    """Master grader — routes to the correct grader based on task_id."""
    if task_id == "task1":
        return grade_task1(
            label=action.get("label"),
            correct_label=task_data["correct_label"],
        )
    elif task_id == "task2":
        return grade_task2(
            ranking=action.get("ranking"),
            correct_ranking=task_data["correct_ranking"],
        )
    elif task_id == "task3":
        return grade_task3(
            label=action.get("label"),
            ranking=action.get("ranking"),
            reply=action.get("reply"),
            correct_label=task_data["correct_label"],
            correct_ranking=task_data["correct_ranking"],
            correct_reply_keywords=task_data["correct_reply_keywords"],
        )
    return 0.05
