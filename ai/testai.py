import requests


API_URL = "http://127.0.0.1:8000/api/chat"


def test_question(question, language="English"):

    print("\n========================================")
    print("QUESTION")
    print("========================================")
    print(question)

    try:

        response = requests.post(
            API_URL,
            json={
                "message": question,
                "language": language,
                "child_email": "demo@example.com"
            },
            timeout=300
        )

        print("\nSTATUS:", response.status_code)

        response.raise_for_status()

        data = response.json()

        print("\nNERAVU RESPONSE:")
        print(data)

    except Exception as e:

        print("\nERROR:")
        print(e)


# ============================================================
# TEST 1 - GENERAL HEALTH
# ============================================================

test_question(
    "What are the symptoms of dehydration?",
    "English"
)


# ============================================================
# TEST 2 - FIRST AID
# ============================================================

test_question(
    "What should I do for a minor burn?",
    "English"
)


# ============================================================
# TEST 3 - MEDICINE SAFETY
# ============================================================

test_question(
    "How should I take medicines safely?",
    "English"
)


# ============================================================
# TEST 4 - EMERGENCY
# ============================================================

test_question(
    "I have severe chest pain and difficulty breathing.",
    "English"
)


# ============================================================
# TEST 5 - KANNADA
# ============================================================

test_question(
    "What are the symptoms of dehydration?",
    "Kannada"
)


# ============================================================
# TEST 6 - HINDI
# ============================================================

test_question(
    "What are the symptoms of dehydration?",
    "Hindi"
)