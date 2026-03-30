"""
Graders for each task. Each returns a float between 0.0 and 1.0.
Deterministic and reproducible — same inputs always give same score.
"""

from typing import List, Optional


def grade_task1(label: Optional[str], correct_label: str) -> float:
    """
    Task 1 (easy): Label the email.
    - Correct label → 1.0
    - Wrong label   → 0.0
    """
    if label is None:
        return 0.0
    return 1.0 if label.strip().lower() == correct_label.lower() else 0.0


def grade_task2(ranking: Optional[List[str]], correct_ranking: List[str]) -> float:
    """
    Task 2 (medium): Prioritize 5 emails.
    Partial credit based on how many emails are in the correct position.
    Score = number of correct positions / total emails
    """
    if not ranking:
        return 0.0

    n = len(correct_ranking)
    if len(ranking) != n:
        # Partial credit even if length is wrong
        correct = sum(
            1 for i, eid in enumerate(ranking)
            if i < n and eid == correct_ranking[i]
        )
        return round(correct / n, 2)

    correct = sum(1 for i in range(n) if ranking[i] == correct_ranking[i])

    # Bonus: check if top 2 are correct (most critical emails)
    top2_correct = ranking[:2] == correct_ranking[:2]
    bonus = 0.1 if top2_correct else 0.0

    score = (correct / n) + bonus
    return round(min(score, 1.0), 2)


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
    Combined score:
      - Label:    30% of total
      - Ranking:  30% of total
      - Reply:    40% of total (keyword coverage)
    """
    # Label score (30%)
    label_score = grade_task1(label, correct_label)

    # Ranking score (30%)
    ranking_score = grade_task2(ranking, correct_ranking)

    # Reply score (40%) — check keyword coverage
    if not reply:
        reply_score = 0.0
    else:
        reply_lower = reply.lower()
        matched = sum(1 for kw in correct_reply_keywords if kw in reply_lower)
        reply_score = matched / len(correct_reply_keywords)

    total = (label_score * 0.30) + (ranking_score * 0.30) + (reply_score * 0.40)
    return round(total, 2)


def grade(task_id: str, action: dict, task_data: dict) -> float:
    """
    Master grader — routes to the correct grader based on task_id.
    action: dict with keys label, ranking, reply
    task_data: the task definition from tasks.py
    """
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
    return 0.0
