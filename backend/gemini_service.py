import os
import tempfile
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------
# 1. SPEECH → TEXT
# ---------------------------------------------------------

def transcribe_audio(audio_bytes: bytes, extension=".wav"):
    """
    Converts user's speech into text using Gemini 3.5 Transcribe.
    Automatically detects English, Kannada, Hindi, Tamil,
    Telugu, Marathi, etc.
    """

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # Upload audio to Gemini
        audio_file = client.files.upload(
            file=temp_path
        )

        # Gemini transcription
        interaction = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[
                {
                    "type": "audio",
                    "uri": audio_file.uri,
                    "mime_type": audio_file.mime_type,
                }
            ],
            generation_config={
                "transcription_config": {
                    # Empty = automatic language detection
                    "language_codes": [],

                    # Smart transcription removes unnecessary
                    # filler words and improves formatting.
                    "mode": "smart"
                }
            }
        )

        transcript = interaction.output_text.strip()

        return transcript

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# 2. TEXT → NERAVU RESPONSE
# ---------------------------------------------------------

def generate_neravu_response(
    user_message: str,
    language: str = "English",
    context: str = ""
):
    """
    Generates Neravu's response using Gemini.
    """

    system_instruction = f"""
You are Neravu, an AI elderly-care companion.

The user's preferred language is: {language}

IMPORTANT RULES:

1. Reply ONLY in {language}.
2. Do not switch to another language.
3. Use simple words that an elderly person can understand.
4. Keep the response short and clear.
5. Do not diagnose diseases.
6. Do not prescribe medicines.
7. Do not invent medical facts.
8. For serious symptoms, recommend contacting a doctor
   or emergency services.
9. If the situation appears immediately dangerous,
   clearly tell the user to seek emergency help.
10. Be calm, respectful and reassuring.
11. Do not use emojis.
12. Prefer 2-4 short sentences.
"""

    if context:
        prompt = f"""
{system_instruction}

Relevant healthcare information:

{context}

User said:

{user_message}

Respond appropriately in {language}.
"""
    else:
        prompt = f"""
{system_instruction}

User said:

{user_message}

Respond appropriately in {language}.
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return interaction.output_text.strip()