import requests


# =========================================================
# BACKEND URL
# =========================================================

API_URL = "https://neravu-new.onrender.com"


# =========================================================
# SEND MESSAGE
# =========================================================

def send_message(
    message,
    language="English"
):

    if not message or not message.strip():

        return {
            "success": False,
            "reply": "Please enter a message."
        }

    payload = {
        "message": message.strip(),
        "language": language
    }

    try:

        response = requests.post(
            f"{API_URL}/api/chat",
            json=payload,
            timeout=600
        )

    except requests.exceptions.ConnectionError:

        raise Exception(
            "Neravu backend is not running. "
            "Please try again in a moment."
        )

    except requests.exceptions.Timeout:

        raise Exception(
            "Neravu backend took too long to respond."
        )

    except Exception as e:

        raise Exception(
            f"Connection error: {e}"
        )

    # -----------------------------------------------------
    # HTTP ERROR
    # -----------------------------------------------------

    if response.status_code != 200:

        try:

            data = response.json()

            error = data.get(
                "error",
                response.text
            )

        except Exception:

            error = response.text

        raise Exception(
            f"Backend error "
            f"{response.status_code}: {error}"
        )

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    try:

        data = response.json()

    except Exception:

        raise Exception(
            "Backend returned invalid JSON."
        )

    # -----------------------------------------------------
    # BACKEND ERROR
    # -----------------------------------------------------

    if not data.get("success", True):

        raise Exception(
            data.get(
                "error",
                data.get(
                    "reply",
                    "Neravu could not generate a response."
                )
            )
        )

    return data