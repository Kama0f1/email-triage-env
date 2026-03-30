"""
3 tasks with increasing difficulty.
Each task has sample emails and correct answers for grading.
"""

TASKS = {
    "task1": {
        "id": "task1",
        "difficulty": "easy",
        "description": (
            "Label the email correctly. "
            "Choose one of: spam, work, personal, urgent."
        ),
        "email": {
            "id": "e1",
            "subject": "CONGRATULATIONS! You won $1,000,000!!!",
            "sender": "winner@totallylegit-prizes.com",
            "body": (
                "Dear Lucky Winner, You have been selected to receive "
                "$1,000,000. Click here to claim your prize immediately. "
                "Provide your bank details to transfer the funds."
            ),
        },
        "correct_label": "spam",
        "correct_ranking": None,
        "correct_reply_keywords": None,
    },

    "task2": {
        "id": "task2",
        "difficulty": "medium",
        "description": (
            "Prioritize these 5 emails from highest to lowest urgency. "
            "Return a ranked list of email IDs."
        ),
        "email": None,
        "emails": [
            {
                "id": "e1",
                "subject": "Server is DOWN — production outage",
                "sender": "alerts@company.com",
                "body": "Critical: The main production server is unreachable. Customers affected.",
            },
            {
                "id": "e2",
                "subject": "Team lunch next Friday?",
                "sender": "colleague@company.com",
                "body": "Hey, want to grab lunch with the team next Friday?",
            },
            {
                "id": "e3",
                "subject": "Invoice overdue — payment required today",
                "sender": "billing@vendor.com",
                "body": "Your invoice #4521 is 30 days overdue. Pay today to avoid service interruption.",
            },
            {
                "id": "e4",
                "subject": "Monthly newsletter — April edition",
                "sender": "newsletter@industry.com",
                "body": "Read our latest industry updates, trends, and tips for this month.",
            },
            {
                "id": "e5",
                "subject": "Meeting rescheduled to tomorrow 9am",
                "sender": "manager@company.com",
                "body": "Hi, I had to move our 1:1 to tomorrow at 9am. Please confirm.",
            },
        ],
        # Correct order: server outage > invoice overdue > meeting rescheduled > lunch > newsletter
        "correct_label": None,
        "correct_ranking": ["e1", "e3", "e5", "e2", "e4"],
        "correct_reply_keywords": None,
    },

    "task3": {
        "id": "task3",
        "difficulty": "hard",
        "description": (
            "You must do 3 things: "
            "(1) label the email as spam/work/personal/urgent, "
            "(2) rank all 3 emails in the inbox by urgency, "
            "(3) draft a professional reply to the urgent email."
        ),
        "email": {
            "id": "e1",
            "subject": "Action required: Contract renewal deadline is tomorrow",
            "sender": "legal@partnercompany.com",
            "body": (
                "Dear Team, This is a reminder that our contract renewal "
                "deadline is tomorrow, April 1st. Please review the attached "
                "document and confirm your acceptance or send proposed changes "
                "by end of business today. Failure to respond will result in "
                "automatic contract termination."
            ),
        },
        "emails": [
            {
                "id": "e1",
                "subject": "Action required: Contract renewal deadline is tomorrow",
                "sender": "legal@partnercompany.com",
                "body": "Contract renewal deadline is tomorrow. Please confirm.",
            },
            {
                "id": "e2",
                "subject": "Great working with you!",
                "sender": "friend@gmail.com",
                "body": "Hey! Really enjoyed collaborating on the last project. Let's do it again!",
            },
            {
                "id": "e3",
                "subject": "Your order has shipped",
                "sender": "noreply@shop.com",
                "body": "Your order #8823 has been shipped. Expected delivery: 3-5 days.",
            },
        ],
        "correct_label": "urgent",
        "correct_ranking": ["e1", "e2", "e3"],
        "correct_reply_keywords": [
            "contract", "confirm", "review", "deadline", "renewal"
        ],
    },
}
