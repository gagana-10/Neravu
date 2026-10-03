import os
import smtplib
from email.message import EmailMessage
from datetime import datetime


def send_child_alert(parent_message, child_email):

    sender_email = os.getenv("NERAVU_EMAIL")
    app_password = os.getenv("NERAVU_APP_PASSWORD")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert = f"""
NERAVU EMERGENCY ALERT

Time: {timestamp}

The elderly user may need immediate assistance.

Reported message:
{parent_message}

Please check on them immediately or contact local emergency medical services.
"""

    try:
        email = EmailMessage()
        email["Subject"] = "🚨 Neravu Emergency Alert"
        email["From"] = sender_email
        email["To"] = child_email
        email.set_content(alert)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(email)

        return {
            "sent": True,
            "recipient": child_email,
            "message": alert
        }

    except Exception as error:

        print("Email alert failed:", error)

        return {
            "sent": False,
            "recipient": child_email,
            "message": alert,
            "error": str(error)
        }