# ============================================================
# NERAVU BACKEND
# FastAPI + Gemini Transcription + Gemini AI
# ============================================================

import os
import sys
import time
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ============================================================
# WINDOWS UTF-8 CONSOLE
# ============================================================

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        pass


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Create backend/.env and add GEMINI_API_KEY=your_key"
    )


# Google Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Gemini model used for speech → text
TRANSCRIPTION_MODEL = "gemini-3.5-transcribe"

# Gemini model used for Neravu responses
CHAT_MODEL = "gemini-3.8-flash"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Neravu AI Backend",
    version="2.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED = [
    "en",
    "kn",
    "hi",
    "ta",
    "te",
    "mr"
]


LANGUAGE_CODES = {

    # English
    "English": "en",

    # Indian languages
    "Kannada": "kn",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",

    # Native UI labels
    "ಕನ್ನಡ": "kn",
    "हिन्दी": "hi",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "मराठी": "mr",

    # Already-code values
    "en": "en",
    "kn": "kn",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "mr": "mr",
}


LANGUAGE_NAMES = {

    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",

}


# ============================================================
# OPTIONAL RAG
# ============================================================

# Your existing project may already contain:
#
#     rag.py
#
# with:
#
#     retrieve_context(message)
#
# We try to import it, but Neravu will still work if it is
# temporarily unavailable.

try:
    from rag import retrieve_context

    RAG_AVAILABLE = True

    print("RAG module loaded successfully.")

except Exception as e:

    RAG_AVAILABLE = False

    print(
        "RAG module not loaded:",
        str(e)
    )

    def retrieve_context(message):
        return ""


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Neravu backend is running",
        "status": "success",
        "transcription_model": TRANSCRIPTION_MODEL,
        "chat_model": CHAT_MODEL
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "voice_ready": True,
        "gemini_configured": bool(GEMINI_API_KEY),
        "transcription_model": TRANSCRIPTION_MODEL,
        "chat_model": CHAT_MODEL,
        "rag_available": RAG_AVAILABLE
    }


# ============================================================
# LANGUAGE NORMALIZATION
# ============================================================

def normalize_language(
    language: str,
    default: str = "en"
) -> str:

    if not language:
        return default

    return LANGUAGE_CODES.get(
        language.strip(),
        default
    )


# ============================================================
# EMERGENCY SAFETY
# ============================================================

EMERGENCY_KEYWORDS = [

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    "chest pain",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "shortness of breath",
    "unconscious",
    "fainted",
    "severe bleeding",
    "heavy bleeding",
    "stroke",
    "heart attack",

    # --------------------------------------------------------
    # Kannada
    # --------------------------------------------------------

    "ಎದೆ ನೋವು",
    "ಉಸಿರಾಟದ ತೊಂದರೆ",
    "ಉಸಿರಾಡಲು ಆಗುತ್ತಿಲ್ಲ",
    "ಪ್ರಜ್ಞೆ ತಪ್ಪಿದೆ",

    # --------------------------------------------------------
    # Hindi
    # --------------------------------------------------------

    "सीने में दर्द",
    "सांस लेने में तकलीफ",
    "सांस नहीं आ रही",
    "बेहोश",

    # --------------------------------------------------------
    # Tamil
    # --------------------------------------------------------

    "மார்பு வலி",
    "மூச்சு விடுவதில் சிரமம்",
    "மூச்சு விட முடியவில்லை",
    "மயக்கம்",

    # --------------------------------------------------------
    # Telugu
    # --------------------------------------------------------

    "ఛాతీ నొప్పి",
    "శ్వాస తీసుకోవడంలో ఇబ్బంది",
    "శ్వాస తీసుకోలేకపోతున్నాను",
    "స్పృహ తప్పింది",

    # --------------------------------------------------------
    # Marathi
    # --------------------------------------------------------

    "छातीत दुखत आहे",
    "श्वास घेण्यास त्रास",
    "श्वास घेता येत नाही",
    "बेशुद्ध",
]


def check_safety(message: str) -> bool:

    if not message:
        return False

    text = message.lower()

    return any(
        keyword.lower() in text
        for keyword in EMERGENCY_KEYWORDS
    )


# ============================================================
# EMERGENCY RESPONSES
# ============================================================

EMERGENCY_RESPONSES = {

    "en":
        "This may be an emergency. Please call emergency services or ask someone nearby to get medical help immediately.",

    "kn":
        "ಇದು ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಾಗಿರಬಹುದು. ದಯವಿಟ್ಟು ತಕ್ಷಣ ತುರ್ತು ಸೇವೆಗಳಿಗೆ ಕರೆ ಮಾಡಿ ಅಥವಾ ಹತ್ತಿರದಲ್ಲಿರುವವರ ಸಹಾಯ ಪಡೆಯಿರಿ.",

    "hi":
        "यह आपातकालीन स्थिति हो सकती है। कृपया तुरंत आपातकालीन सेवा को कॉल करें या पास के किसी व्यक्ति से चिकित्सा सहायता लेने को कहें।",

    "ta":
        "இது அவசரநிலையாக இருக்கலாம். தயவுசெய்து உடனடியாக அவசர சேவையை அழைக்கவும் அல்லது அருகிலுள்ள ஒருவரின் உதவியைப் பெறவும்.",

    "te":
        "ఇది అత్యవసర పరిస్థితి కావచ్చు. దయచేసి వెంటనే అత్యవసర సేవలకు కాల్ చేయండి లేదా దగ్గరలో ఉన్నవారి సహాయం తీసుకోండి.",

    "mr":
        "ही आपत्कालीन परिस्थिती असू शकते. कृपया त्वरित आपत्कालीन सेवांना कॉल करा किंवा जवळच्या व्यक्तीची वैद्यकीय मदत घ्या.",
}


def emergency_response(language: str) -> str:

    return EMERGENCY_RESPONSES.get(
        language,
        EMERGENCY_RESPONSES["en"]
    )


# ============================================================
# GEMINI SYSTEM INSTRUCTION
# ============================================================

def build_system_instruction(
    language: str,
    context: str = ""
) -> str:

    language_name = LANGUAGE_NAMES.get(
        language,
        "English"
    )

    instruction = f"""
You are Neravu, a safe and friendly AI companion
for elderly people in India.

The user's preferred response language is:
{language_name}

IMPORTANT LANGUAGE RULES:

1. Reply ONLY in {language_name}.
2. Never switch to Hindi, English, Kannada,
   Tamil, Telugu, or Marathi unless the requested
   response language is that language.
3. Do not translate the user's message unless
   translation is explicitly requested.
4. Answer the user's actual question.
5. Use natural, simple language suitable for
   an elderly person.
6. Keep the response short: approximately
   1 to 3 sentences.
7. Do not repeat the user's message.
8. Do not explain what the user said.
9. Do not diagnose diseases.
10. Do not prescribe medicines or dosages.
11. Do not invent medical information.
12. Give general safe health guidance.
13. If symptoms appear serious, recommend
    contacting a doctor.
14. If symptoms indicate an emergency, tell
    the user to seek emergency medical help.
15. Be calm, respectful and reassuring.
16. Do not use emojis.
17. Do not use complicated medical terminology
    unless necessary.
"""

    if context:

        instruction += f"""

RELEVANT HEALTHCARE INFORMATION:

{context}

Use the healthcare information above when
it is relevant to the user's question.

Do not invent information that is not supported
by the provided healthcare information.
"""

    return instruction.strip()


# ============================================================
# GEMINI CHAT RESPONSE
# ============================================================

def generate_gemini_response(
    message: str,
    language: str,
    context: str = ""
):

    system_instruction = build_system_instruction(
        language,
        context
    )

    language_name = LANGUAGE_NAMES.get(
        language,
        "English"
    )

    prompt = f"""
The user has spoken or typed the following:

{message}

Respond directly to the user.

Your response MUST be in:
{language_name}

Remember:
- Do not repeat the user's message.
- Do not translate the user's message.
- Answer the user's actual concern.
- Keep it short and easy for an elderly person.
"""

    try:

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                system_instruction=system_instruction,

                temperature=0.2,

                max_output_tokens=300
            )
        )

        answer = (response.text or "").strip()

        if not answer:

            return None

        return answer

    except Exception as e:

        print(
            "Gemini chat error:",
            str(e)
        )

        return None


# ============================================================
# RAG + GEMINI RESPONSE
# ============================================================

def generate_reply(
    message: str,
    language: str
):

    debug = {

        "model": CHAT_MODEL,

        "language": language,

        "rag_used": False,

        "emergency": False
    }

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if check_safety(message):

        debug["emergency"] = True

        return (
            emergency_response(language),
            True,
            debug
        )

    # --------------------------------------------------------
    # Retrieve relevant healthcare context
    # --------------------------------------------------------

    context = ""

    if RAG_AVAILABLE:

        try:

            context = retrieve_context(
                message
            )

            if context:

                debug["rag_used"] = True

        except Exception as e:

            print(
                "RAG error:",
                str(e)
            )

            context = ""

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    answer = generate_gemini_response(
        message=message,
        language=language,
        context=context
    )

    if answer:

        return (
            answer,
            False,
            debug
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    fallback = {

        "en":
            "Sorry, Neravu could not answer right now. Please try again.",

        "kn":
            "ಕ್ಷಮಿಸಿ, ನೆರವು ಈಗ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",

        "hi":
            "क्षमा करें, नेरवु अभी उत्तर नहीं दे पा रहा है। कृपया फिर से प्रयास करें।",

        "ta":
            "மன்னிக்கவும், நெரவு இப்போது பதிலளிக்க முடியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",

        "te":
            "క్షమించండి, నెరవు ఇప్పుడు సమాధానం ఇవ్వలేకపోతోంది. దయచేసి మళ్లీ ప్రయత్నించండి.",

        "mr":
            "क्षमस्व, नेरवु आत्ता उत्तर देऊ शकत नाही. कृपया पुन्हा प्रयत्न करा."
    }

    return (
        fallback.get(
            language,
            fallback["en"]
        ),
        False,
        debug
    )


# ============================================================
# CHAT RESPONSE FORMAT
# ============================================================

def _chat_result(
    answer,
    urgent,
    language_code,
    success=True,
    error=None,
    debug=None
):

    result = {

        "success": success,

        "response": answer,

        # Keep this because your Streamlit frontend
        # already reads "reply".
        "reply": answer,

        "urgent": urgent,

        "language": language_code
    }

    if error:

        result["error"] = error

    if debug:

        result["debug"] = debug

    return result


# ============================================================
# CHAT API
# ============================================================

@app.post("/api/chat")
def chat(request_data: dict):

    try:

        message = (
            request_data.get(
                "message",
                ""
            )
            or ""
        ).strip()

        language = request_data.get(
            "language",
            "English"
        )

        # ----------------------------------------------------
        # Empty message
        # ----------------------------------------------------

        if not message:

            return _chat_result(

                "Please say something.",

                False,

                "en",

                success=False,

                error="Empty message"
            )

        # ----------------------------------------------------
        # Normalize language
        # ----------------------------------------------------

        language_code = normalize_language(
            language
        )

        print()
        print("------------------------------------------")
        print("CHAT REQUEST")
        print("Message :", message)
        print(
            "Language:",
            language,
            "->",
            language_code
        )
        print("------------------------------------------")

        # ----------------------------------------------------
        # Generate response
        # ----------------------------------------------------

        answer, urgent, debug = generate_reply(
            message,
            language_code
        )

        return _chat_result(

            answer,

            urgent,

            language_code,

            debug=debug
        )

    except Exception as e:

        print(
            "Chat API error:",
            str(e)
        )

        return _chat_result(

            "Sorry, something went wrong.",

            False,

            "en",

            success=False,

            error=str(e)
        )


# ============================================================
# OLD CHAT ENDPOINT
# ============================================================

# Your older frontend may still call /chat.
# Keep this route so nothing breaks.

@app.post("/chat")
def old_chat_endpoint(
    request_data: dict
):

    return chat(
        request_data
    )


# ============================================================
# AUDIO MIME TYPE
# ============================================================

def get_mime_type(
    filename: str
) -> str:

    extension = Path(
        filename or ""
    ).suffix.lower()

    mime_types = {

        ".wav": "audio/wav",

        ".mp3": "audio/mpeg",

        ".mpeg": "audio/mpeg",

        ".mp4": "audio/mp4",

        ".m4a": "audio/mp4",

        ".webm": "audio/webm",

        ".ogg": "audio/ogg",

        ".oga": "audio/ogg",

        ".flac": "audio/flac",

        ".aac": "audio/aac",

        ".opus": "audio/ogg"
    }

    return mime_types.get(
        extension,
        "audio/wav"
    )


# ============================================================
# GEMINI AUDIO TRANSCRIPTION
# ============================================================

def transcribe_with_gemini(
    audio_bytes: bytes,
    filename: str,
    language_hint: str = None,
    force_language: bool = False
):

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary audio file
        # ----------------------------------------------------

        extension = (
            Path(
                filename or ""
            ).suffix.lower()
        )

        if not extension:

            extension = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(
                audio_bytes
            )

            temp_path = temp_file.name

        mime_type = get_mime_type(
            filename
        )

        print()
        print("==========================================")
        print("GEMINI VOICE REQUEST")
        print("Audio size :", len(audio_bytes), "bytes")
        print("Filename   :", filename)
        print("MIME type  :", mime_type)
        print("Hint       :", language_hint)
        print("Forced     :", force_language)
        print("==========================================")

        # ----------------------------------------------------
        # Upload audio to Gemini Files API
        # ----------------------------------------------------

        audio_file = client.files.upload(
            file=temp_path
        )

        # ----------------------------------------------------
        # Language configuration
        # ----------------------------------------------------

        transcription_config = {

            # Smart transcription gives cleaner output.
            "mode": "smart"
        }

        # If the app explicitly forces a language,
        # give Gemini the selected language code.
        #
        # Otherwise leave language_codes empty so Gemini
        # automatically detects the spoken language.
        if force_language and language_hint:

            transcription_config[
                "language_codes"
            ] = [language_hint]

        else:

            transcription_config[
                "language_codes"
            ] = []

        # ----------------------------------------------------
        # Gemini transcription
        # ----------------------------------------------------

        interaction = client.interactions.create(

            model=TRANSCRIPTION_MODEL,

            input=[

                {
                    "type": "audio",

                    "uri": audio_file.uri,

                    "mime_type": audio_file.mime_type
                    or mime_type
                }

            ],

            generation_config={

                "transcription_config":
                    transcription_config
            }
        )

        text = (
            interaction.output_text
            or ""
        ).strip()

        # ----------------------------------------------------
        # Clean repeated whitespace
        # ----------------------------------------------------

        text = " ".join(
            text.split()
        )

        print()
        print("Gemini transcription:")
        print(text)
        print()

        if not text:

            return None, None

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Gemini Transcribe does not require us to calculate
        # Whisper-style probabilities.
        #
        # If the user selected a language, we use that as the
        # application language for the response.
        #
        # Otherwise the UI/backend can use the selected app
        # language as the response language.
        # ----------------------------------------------------

        detected_language = (
            language_hint
            if language_hint
            else None
        )

        return text, detected_language

    finally:

        # ----------------------------------------------------
        # Delete local temporary file
        # ----------------------------------------------------

        if temp_path:

            try:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )

            except Exception:

                pass


# ============================================================
# ERROR HELPER
# ============================================================

def _fail(
    message: str,
    status: int = 400,
    **extra
):

    body = {

        "success": False,

        "text": "",

        "error": message
    }

    body.update(extra)

    return JSONResponse(
        status_code=status,
        content=body
    )


# ============================================================
# TRANSCRIBE API
# ============================================================

@app.post("/transcribe")
async def transcribe_audio(

    file: UploadFile = File(...),

    language: str = Form(""),

    force: str = Form("0")
):

    try:

        # ----------------------------------------------------
        # Read audio
        # ----------------------------------------------------

        audio_bytes = await file.read()

        if not audio_bytes:

            return _fail(
                "No audio data received.",
                400
            )

        # ----------------------------------------------------
        # Normalize selected language
        # ----------------------------------------------------

        hint = (
            normalize_language(
                language,
                default=None
            )
            if language
            else None
        )

        # ----------------------------------------------------
        # Force mode
        # ----------------------------------------------------

        forced = (
            force == "1"
            and bool(hint)
        )

        # ----------------------------------------------------
        # Gemini transcription
        # ----------------------------------------------------

        start_time = time.time()

        text, detected_language = (
            transcribe_with_gemini(

                audio_bytes=audio_bytes,

                filename=file.filename or "audio.wav",

                language_hint=hint,

                force_language=forced
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Gemini transcription took "
            f"{elapsed:.2f} seconds"
        )

        # ----------------------------------------------------
        # Empty transcription
        # ----------------------------------------------------

        if not text:

            return _fail(
                "Could not understand the audio. "
                "Please speak again closer to the microphone.",
                400,
                language=hint
            )

        # ----------------------------------------------------
        # Language used for Neravu response
        # ----------------------------------------------------

        language_used = (
            hint
            if hint
            else detected_language
        )

        if not language_used:

            language_used = "en"

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {

            "success": True,

            "text": text,

            "language": language_used,

            "detected_language":
                detected_language,

            "forced": forced,

            "low_confidence": False,

            "warning": None,

            "hint_mismatch": False,

            "app_language": hint,

            "model":
                TRANSCRIPTION_MODEL
        }

    except Exception as e:

        print()
        print("==========================================")
        print("GEMINI TRANSCRIPTION ERROR")
        print(str(e))
        print("==========================================")
        print()

        return _fail(
            str(e),
            500
        )


# ============================================================
# VOICE CHAT API
# ============================================================

@app.post("/api/voice-chat")
async def voice_chat(

    file: UploadFile = File(...),

    language: str = Form("English"),

    force: str = Form("0")
):

    try:

        # ----------------------------------------------------
        # Read audio
        # ----------------------------------------------------

        audio_bytes = await file.read()

        if not audio_bytes:

            return _fail(
                "No audio data received.",
                400
            )

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        language_code = normalize_language(
            language
        )

        # ----------------------------------------------------
        # Transcribe
        # ----------------------------------------------------

        start_time = time.time()

        transcript, _ = (
            transcribe_with_gemini(

                audio_bytes=audio_bytes,

                filename=file.filename or "audio.wav",

                language_hint=language_code,

                force_language=(
                    force == "1"
                )
            )
        )

        transcription_time = (
            time.time()
            - start_time
        )

        if not transcript:

            return _fail(
                "Could not understand the audio. "
                "Please record again.",
                400
            )

        print(
            "Voice transcript:",
            transcript
        )

        # ----------------------------------------------------
        # Generate Neravu response
        # ----------------------------------------------------

        answer, urgent, debug = generate_reply(

            transcript,

            language_code
        )

        return {

            "success": True,

            "transcript":
                transcript,

            "text":
                transcript,

            "response":
                answer,

            "reply":
                answer,

            "urgent":
                urgent,

            "language":
                language_code,

            "transcription_time":
                round(
                    transcription_time,
                    2
                ),

            "debug":
                debug
        }

    except Exception as e:

        print(
            "Voice chat error:",
            str(e)
        )

        return _fail(
            str(e),
            500
        )


# ============================================================
# STARTUP MESSAGE
# ============================================================

print()
print("==========================================")
print("        NERAVU GEMINI BACKEND")
print("==========================================")
print(
    "Transcription model:",
    TRANSCRIPTION_MODEL
)
print(
    "Chat model         :",
    CHAT_MODEL
)
print(
    "Gemini API key     :",
    "CONFIGURED"
    if GEMINI_API_KEY
    else "MISSING"
)
print(
    "RAG                :",
    "AVAILABLE"
    if RAG_AVAILABLE
    else "NOT AVAILABLE"
)
print("==========================================")
print()