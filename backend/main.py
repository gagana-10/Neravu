# ============================================================
# NERAVU BACKEND
# FastAPI + Whisper + Ollama
# ============================================================

import os
import shutil
import tempfile
import requests
import sys
import time

# Neravu transcribes Kannada, Hindi, Tamil, Telugu and Marathi, and every
# request logs the text it produced. On Windows the console is cp1252, so
# print()ing that text raised UnicodeEncodeError ("charmap codec can't
# encode characters..."), the handler caught it, and a perfectly good
# transcription came back to the user as a 500. Logging must never be able
# to fail a request.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import whisper

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI(title="Neravu AI Backend", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Model used ONLY for translation (the one your translator.py already uses).
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "translategemma:4b")

# True  : user text -> English -> Qwen answers in English -> translated to the spoken language
#         (most reliable for Kannada/Telugu/Tamil/Marathi)
# False : Qwen answers directly in the target language (faster, less reliable)
USE_TRANSLATION_PIPELINE = os.getenv("USE_TRANSLATION_PIPELINE", "1") == "1"

# "small" is weak for Kannada and writes it as Devanagari gibberish, so the
# default is "medium" on every machine, GPU or not.
# Faster but much less accurate:  set WHISPER_MODEL=small
# Best accuracy (needs ~10 GB RAM/VRAM):  set WHISPER_MODEL=large-v3
HAS_GPU = torch.cuda.is_available()
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")

# If the user picked an Indian language in the app and Whisper's top guess is a
# DIFFERENT Indian language, we trust the user's choice as long as Whisper gave
# it at least this fraction of the top guess's probability.
# (Fixes Kannada being detected as Hindi/Telugu. English is never overridden.)
HINT_BIAS = 0.25

# Whisper spreads its language probabilities over all 99 languages it knows.
# If the best of OUR six scores below this, the clip is almost certainly
# silence, background noise, or a language Neravu does not support. Picking
# the argmax of six near-zero floats is a coin toss, which is how a silent
# recording used to come back as a confident language detection.
MIN_LANG_CONFIDENCE = 0.10

# Raw speech-to-text is held to a looser script standard than generated text.
# Elderly Indian speech routinely mixes in English words - "sugar", "BP",
# "tablet", "doctor" - and transcribing those in Latin letters is CORRECT,
# not a wrong-script error. At the old 0.9 threshold such a transcription was
# rejected and the user was told it had been misunderstood.
#
# Measured on sample Kannada sentences:
#     pure Kannada ................................ 1.00
#     one English word mixed in ................... 0.75
#     two English words mixed in .................. 0.54
#     WRONG - Devanagari written for Kannada ...... 0.00
#     WRONG - Latin transliteration ............... 0.00
# A genuinely wrong script scores zero, so anything in (0.00, 0.54] splits
# the two cases. 0.50 keeps margin on both sides.
ASR_SCRIPT_THRESHOLD = 0.50

# Peak amplitude across the whole clip. Below this, nobody actually spoke:
# Whisper hallucinates confident sentences out of silence.
SILENCE_PEAK = 0.01

# ffmpeg is NOT a Python package - Whisper shells out to the binary to decode
# audio. Resolve it once at start-up so a missing install is reported plainly
# instead of surfacing as a bare WinError 2 from inside whisper.load_audio().
FFMPEG_PATH = shutil.which("ffmpeg")


# ============================================================
# LANGUAGE CONFIGURATION
# ============================================================

SUPPORTED = ["en", "kn", "hi", "ta", "te", "mr"]

LANGUAGE_CODES = {
    "English": "en", "Kannada": "kn", "Hindi": "hi",
    "Tamil": "ta", "Telugu": "te", "Marathi": "mr",

    # native labels (in case the UI ever sends them)
    "ಕನ್ನಡ": "kn", "हिन्दी": "hi", "தமிழ்": "ta",
    "తెలుగు": "te", "मराठी": "mr",

    "en": "en", "kn": "kn", "hi": "hi",
    "ta": "ta", "te": "te", "mr": "mr",
}

LANGUAGE_NAMES = {
    "en": "English", "kn": "Kannada", "hi": "Hindi",
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
}


# ============================================================
# LOAD WHISPER
# ============================================================

print("==========================================")
print(f"Loading Whisper model: {WHISPER_MODEL_NAME}  (GPU: {HAS_GPU})")
print("==========================================")

try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    print("Whisper model loaded successfully!")
except Exception as e:
    whisper_model = None
    print("ERROR LOADING WHISPER:", str(e))

if FFMPEG_PATH:
    print("ffmpeg:", FFMPEG_PATH)
else:
    print("WARNING: ffmpeg was NOT found on PATH.")
    print("         Whisper cannot decode audio without it, so every")
    print("         /transcribe request will fail. Install it with:")
    print("             winget install Gyan.FFmpeg     (Windows)")
    print("             brew install ffmpeg            (macOS)")
    print("             sudo apt install ffmpeg        (Debian/Ubuntu)")


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "Neravu backend is running", "status": "success"}


@app.get("/health")
def health():
    voice_ready = whisper_model is not None and FFMPEG_PATH is not None
    return {
        # "ok" used to be reported even with no Whisper and no ffmpeg, which
        # made the health check useless for diagnosing a dead voice feature.
        "status": "ok" if voice_ready else "degraded",
        "voice_ready": voice_ready,
        "whisper_loaded": whisper_model is not None,
        "whisper_model": WHISPER_MODEL_NAME,
        "ffmpeg": FFMPEG_PATH,
        "ollama_model": OLLAMA_MODEL,
    }


def normalize_language(language: str, default: str = "en") -> str:
    if not language:
        return default
    return LANGUAGE_CODES.get(language.strip(), default)


# ============================================================
# SAFETY CHECK
# ============================================================

EMERGENCY_KEYWORDS = [
    # English
    "chest pain", "difficulty breathing", "can't breathe", "cannot breathe",
    "shortness of breath", "unconscious", "fainted", "severe bleeding",
    "heavy bleeding", "stroke", "heart attack",
    # Kannada
    "ಎದೆ ನೋವು", "ಉಸಿರಾಟದ ತೊಂದರೆ", "ಉಸಿರಾಡಲು ಆಗುತ್ತಿಲ್ಲ", "ಪ್ರಜ್ಞೆ ತಪ್ಪಿದೆ",
    # Hindi
    "सीने में दर्द", "सांस लेने में तकलीफ", "सांस नहीं आ रही", "बेहोश",
    # Tamil
    "மார்பு வலி", "மூச்சு விடுவதில் சிரமம்", "மூச்சு விட முடியவில்லை", "மயக்கம்",
    # Telugu
    "ఛాతీ నొప్పి", "శ్వాస తీసుకోవడంలో ఇబ్బంది",
    "శ్వాస తీసుకోలేకపోతున్నాను", "స్పృహ తప్పింది",
    # Marathi
    "छातीत दुखत आहे", "श्वास घेण्यास त्रास", "श्वास घेता येत नाही", "बेशुद्ध",
]


def check_safety(message: str) -> bool:
    text = message.lower()
    return any(k in text for k in EMERGENCY_KEYWORDS)


EMERGENCY_RESPONSES = {
    "en": ("This may be an emergency. Please call emergency services or ask "
           "someone nearby to get medical help immediately."),
    "kn": ("ಇದು ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಾಗಿರಬಹುದು. ದಯವಿಟ್ಟು ತಕ್ಷಣ ತುರ್ತು ಸೇವೆಗಳಿಗೆ ಕರೆ ಮಾಡಿ ಅಥವಾ "
           "ಹತ್ತಿರದಲ್ಲಿರುವವರ ಸಹಾಯ ಪಡೆಯಿರಿ."),
    "hi": ("यह आपातकालीन स्थिति हो सकती है। कृपया तुरंत आपातकालीन सेवा को कॉल करें या "
           "पास के किसी व्यक्ति से चिकित्सा सहायता लेने को कहें।"),
    "ta": ("இது அவசரநிலையாக இருக்கலாம். தயவுசெய்து உடனடியாக அவசர சேவையை அழைக்கவும் "
           "அல்லது அருகிலுள்ள ஒருவரின் உதவியைப் பெறவும்."),
    "te": ("ఇది అత్యవసర పరిస్థితి కావచ్చు. దయచేసి వెంటనే అత్యవసర సేవలకు కాల్ చేయండి "
           "లేదా దగ్గరలో ఉన్నవారి సహాయం తీసుకోండి."),
    "mr": ("ही आपत्कालीन परिस्थिती असू शकते. कृपया त्वरित आपत्कालीन सेवांना कॉल करा "
           "किंवा जवळच्या व्यक्तीची वैद्यकीय मदत घ्या."),
}


def emergency_response(language: str) -> str:
    return EMERGENCY_RESPONSES.get(language, EMERGENCY_RESPONSES["en"])


# ============================================================
# OLLAMA
# ============================================================

def build_prompt(message: str, language: str) -> str:
    language_name = LANGUAGE_NAMES.get(language, "English")

    return f"""
You are Neravu, a friendly AI companion for elderly people in India.

USER MESSAGE:
{message}

LANGUAGE:
{language_name}

IMPORTANT:

- Reply ONLY in {language_name}.
- Do NOT repeat the user's message.
- Do NOT translate the user's message.
- Do NOT explain what the user said.
- Answer the user's actual problem.
- Use simple natural {language_name}.
- Keep the response to 1 to 3 sentences.
- Be calm and reassuring.
- Do not use emojis.
- Do not give a medical diagnosis.
- Do not invent medicines or dosages.
- For health problems, provide general safe advice.
- If symptoms sound serious, recommend getting medical help.

Now answer the user.
"""


def generate_ollama_response(message: str, language: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_prompt(message, language),
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 300},
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate", json=payload, timeout=120
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return answer or "Sorry, I could not generate a response."

    except requests.exceptions.ConnectionError:
        return "Neravu's AI service is not running. Please make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "The AI service is taking too long to respond. Please try again."
    except Exception as e:
        print("Ollama error:", str(e))
        return "Sorry, I could not process your request."


# ============================================================
# RELIABLE MULTILINGUAL REPLY PIPELINE
# ============================================================

SCRIPT_RANGES = {
    "kn": (0x0C80, 0x0CFF),
    "te": (0x0C00, 0x0C7F),
    "ta": (0x0B80, 0x0BFF),
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
}

UNAVAILABLE = {
    "en": "Sorry, Neravu could not answer right now. Please make sure Ollama is running.",
    "kn": "ಕ್ಷಮಿಸಿ, ನೆರವು ಈಗ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು Ollama ಚಾಲನೆಯಲ್ಲಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
    "hi": "क्षमा करें, नेरवु अभी उत्तर नहीं दे पा रहा है। कृपया जाँचें कि Ollama चल रहा है।",
    "ta": "மன்னிக்கவும், நெரவு இப்போது பதிலளிக்க முடியவில்லை. Ollama இயங்குகிறதா என்று சரிபார்க்கவும்.",
    "te": "క్షమించండి, నెరవు ఇప్పుడు సమాధానం ఇవ్వలేకపోతోంది. Ollama నడుస్తోందో లేదో చూడండి.",
    "mr": "क्षमस्व, नेरवु आत्ता उत्तर देऊ शकत नाही. कृपया Ollama सुरू आहे का ते तपासा.",
}


REFUSAL_MARKERS = [
    "unable to", "cannot translate", "can't translate", "i'm sorry", "i am sorry",
    "please provide", "could you please", "could you provide", "as an ai",
    "i don't understand", "i do not understand", "not able to", "no text",
    "there is no", "the text you", "clarify",
]


def looks_like_refusal(text: str) -> bool:
    """Catches translator output like 'I am unable to understand this text'."""
    t = text.lower()
    return any(m in t for m in REFUSAL_MARKERS)


def script_ok(text: str, code: str, threshold: float = 0.9) -> bool:
    """True if most letters of `text` are in the script of language `code`."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if code == "en" or code not in SCRIPT_RANGES:
        good = sum(1 for c in letters if c.isascii())
    else:
        lo, hi = SCRIPT_RANGES[code]
        good = sum(1 for c in letters if lo <= ord(c) <= hi)
    return good / len(letters) >= threshold


def _ollama_generate(prompt, model, num_predict=300, temperature=0.3):
    """Low-level Ollama call. Returns text, or None on any failure."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": temperature, "num_predict": num_predict},
            },
            timeout=240,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip() or None
    except Exception as e:
        print(f"Ollama error ({model}):", str(e))
        return None


def clean_translation(text: str) -> str:
    text = text.strip().strip("\"'\u201c\u201d")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 1 and lines[0].endswith(":"):
        lines = lines[1:]
    return " ".join(lines).strip().strip("\"'\u201c\u201d").strip()


def translate(text: str, src: str, tgt: str):
    """
    Translate with TRANSLATE_MODEL using the prompt format TranslateGemma was
    trained on (single user message, persona line, two blank lines, then text).
    Returns text in the right script, or None.
    """
    src_name, tgt_name = LANGUAGE_NAMES[src], LANGUAGE_NAMES[tgt]

    prompt = (
        f"You are a professional {src_name} ({src}) to {tgt_name} ({tgt}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original "
        f"{src_name} text while adhering to {tgt_name} grammar, vocabulary, and "
        f"cultural sensitivities.\n"
        f"Produce only the {tgt_name} translation, without any additional explanations "
        f"or commentary. Please translate the following {src_name} text into "
        f"{tgt_name}:\n\n\n"
        f"{text}"
    )

    for attempt in range(2):
        out = _ollama_generate(
            prompt, TRANSLATE_MODEL, num_predict=500,
            temperature=0 if attempt == 0 else 0.3,      # retry must differ from try 1
        )
        if not out:
            continue
        out = clean_translation(out)

        if looks_like_refusal(out) and not looks_like_refusal(text):
            print(f"Translation {src}->{tgt} attempt {attempt + 1} was a refusal: {out[:80]!r}")
            continue

        # reject wrong/mixed script, and runaway output far longer than the input
        if out and script_ok(out, tgt) and len(out) <= max(400, len(text) * 8):
            return out
        print(f"Translation {src}->{tgt} attempt {attempt + 1} rejected: {out[:80]!r}")
    return None


def qwen_translate_to_english(text: str, code: str):
    """Fallback understanding step: ask Qwen to put the user's words into English."""
    lang = LANGUAGE_NAMES[code]
    prompt = (
        f"The following {lang} sentence was spoken by an elderly person in India, "
        f"most likely about their health or daily life. Some words may be spelled "
        f"imperfectly because it came from speech recognition.\n"
        f"Translate it into simple English. Output ONLY the English sentence.\n\n"
        f"{lang}: {text}\nEnglish:"
    )
    out = _ollama_generate(prompt, OLLAMA_MODEL, num_predict=120, temperature=0)
    if out:
        out = clean_translation(out)
        if out and script_ok(out, "en") and not looks_like_refusal(out):
            return out
    return None


def direct_reply(message: str, code: str):
    """Ask Qwen to answer straight in the target language; accept only the right script."""
    for _ in range(2):
        out = _ollama_generate(build_prompt(message, code), OLLAMA_MODEL)
        if out and script_ok(out, code):
            return out
        print(f"Direct {code} reply rejected: {(out or '')[:80]!r}")
    return None


def generate_reply(message: str, code: str):
    """
    Returns (reply_text, is_urgent, debug_dict).
    The reply is ALWAYS in language `code` whenever any model can produce it.
    """
    debug = {"path": "english" if code == "en" else "pipeline",
             "english_understanding": None, "english_answer": None}

    # English: simple path
    if code == "en":
        out = _ollama_generate(build_prompt(message, "en"), OLLAMA_MODEL)
        return (out or UNAVAILABLE["en"]), False, debug

    t0 = time.time()

    debug["heard_text"] = message

    # Pipeline: native -> English -> Qwen (English) -> native
    if USE_TRANSLATION_PIPELINE:

        # 1) understand the user's words in English
        if script_ok(message, "en"):
            english_in, stage = message, "typed in English (no translation needed)"
        else:
            english_in = translate(message, code, "en")
            stage = "translategemma"
            if not english_in:
                english_in = qwen_translate_to_english(message, code)
                stage = "qwen fallback"
            if not english_in:
                stage = "FAILED - could not understand the text"

        debug["english_understanding"] = english_in
        debug["input_stage"] = stage
        print(f"English understanding ({stage}):", english_in)

        # 2) answer in English, then translate back
        if english_in:
            if check_safety(english_in):
                return emergency_response(code), True, debug

            english_answer = _ollama_generate(build_prompt(english_in, "en"), OLLAMA_MODEL)
            debug["english_answer"] = english_answer
            print("English answer       :", english_answer)

            if english_answer:
                native = translate(english_answer, "en", code)
                if native:
                    print(f"Pipeline reply took {time.time() - t0:.1f}s")
                    return native, False, debug

    # Fallback: let Qwen answer directly in the target language
    debug["path"] = "direct-fallback"
    out = direct_reply(message, code)
    if out:
        return out, False, debug

    return UNAVAILABLE.get(code, UNAVAILABLE["en"]), False, debug


# ============================================================
# CHAT API
# ============================================================

def _chat_result(answer, urgent, language_code, success=True, error=None, debug=None):
    # "response" is the canonical key; "reply" is kept so the Streamlit
    # frontend (which reads "reply") works either way.
    result = {
        "success": success,
        "response": answer,
        "reply": answer,
        "urgent": urgent,
        "language": language_code,
    }
    if error:
        result["error"] = error
    if debug:
        result["debug"] = debug
    return result


@app.post("/api/chat")
def chat(request_data: dict):
    try:
        message = (request_data.get("message", "") or "").strip()
        language = request_data.get("language", "English")

        if not message:
            return _chat_result("Please say something.", False, "en", success=False,
                                error="Empty message")

        language_code = normalize_language(language)

        print("------------------------------------------")
        print("CHAT REQUEST")
        print("Message :", message)
        print("Language:", language, "->", language_code)
        print("------------------------------------------")

        if check_safety(message):
            return _chat_result(emergency_response(language_code), True, language_code)

        answer, urgent, debug = generate_reply(message, language_code)
        return _chat_result(answer, urgent, language_code, debug=debug)

    except Exception as e:
        print("Chat API error:", str(e))
        return _chat_result("Sorry, something went wrong.", False, "en",
                            success=False, error=str(e))


@app.post("/chat")
def old_chat_endpoint(request_data: dict):
    return chat(request_data)


# ============================================================
# SPOKEN-LANGUAGE DETECTION
# ============================================================

def choose_language(probs: dict, hint: str = None):
    """
    probs : {lang_code: probability} from Whisper (all 99 of its languages)
    hint  : language code the user selected in the app (or None)

    Returns (chosen_code, detected_code, restricted_probs, low_confidence)

    - Only our 6 supported languages are considered.
    - Best Whisper guess wins, EXCEPT when both the guess and the user's
      hint are Indian languages and the hint is a plausible runner-up
      (>= HINT_BIAS * best). English is never overridden in either direction,
      so English speech with Kannada selected still comes out as English.
    - low_confidence is True when even the winner scored under
      MIN_LANG_CONFIDENCE. HINT_BIAS is a RELATIVE test with no floor, so on
      silence or an unsupported language it used to promote a hint sitting at
      a probability of 0.002 and present it as a detection.
    """
    restricted = {c: float(probs.get(c, 0.0)) for c in SUPPORTED}
    detected = max(restricted, key=restricted.get)
    chosen = detected
    low_confidence = restricted[detected] < MIN_LANG_CONFIDENCE

    if (
        hint in restricted
        and hint != detected
        and hint != "en"
        and detected != "en"
        and restricted[hint] >= HINT_BIAS * restricted[detected]
    ):
        chosen = hint

    # Nothing was recognised with any real confidence. Deferring to the
    # language the user actually chose beats transcribing in one picked out
    # of statistical noise.
    if low_confidence and hint in restricted:
        chosen = hint

    return chosen, detected, restricted, low_confidence


def detect_spoken_language(audio, hint: str = None):
    """`audio` is an already-decoded waveform, NOT a path.

    It used to take a path and call whisper.load_audio() itself, which meant
    ffmpeg decoded the same clip two or three times per request.
    """
    n_mels = getattr(whisper_model.dims, "n_mels", 80)
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio), n_mels=n_mels
    ).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    return choose_language(probs, hint)


# ============================================================
# SCRIPT-GUARDED TRANSCRIPTION
# ============================================================

# Used ONLY to steer Whisper when it writes the wrong script (retry pass).
WHISPER_PROMPTS = {
    "kn": "ನಮಸ್ಕಾರ. ನಾನು ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಆರೋಗ್ಯದ ಬಗ್ಗೆ ಸಹಾಯ ಬೇಕು.",
    "hi": "नमस्ते। मैं हिंदी में बोल रहा हूँ। मुझे स्वास्थ्य के बारे में मदद चाहिए।",
    "ta": "வணக்கம். நான் தமிழில் பேசுகிறேன். எனக்கு உடல்நலம் குறித்து உதவி வேண்டும்.",
    "te": "నమస్కారం. నేను తెలుగులో మాట్లాడుతున్నాను. నాకు ఆరోగ్యం గురించి సహాయం కావాలి.",
    "mr": "नमस्कार. मी मराठीत बोलत आहे. मला आरोग्याबद्दल मदत हवी आहे.",
}


def run_whisper(audio, lang, prompt=None):
    result = whisper_model.transcribe(
        audio,
        language=lang,
        task="transcribe",
        fp16=False,
        # fallback temperatures let Whisper retry if it starts repeating itself
        temperature=(0.0, 0.2, 0.4),
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        initial_prompt=prompt,
    )
    return " ".join(result.get("text", "").split()).strip()


def transcribe_with_guard(audio, lang):
    """
    Transcribe in `lang`. If the text comes out in the WRONG SCRIPT
    (e.g. Devanagari letters for Kannada speech), retry once with a short
    prompt written in the right script. Returns (text, warning_or_None).

    Judged at ASR_SCRIPT_THRESHOLD, not the 0.9 used for generated text:
    a transcription like "ಸಕ್ಕರೆ ಕಾಯಿಲೆ ಇದೆ, sugar tablet ತಗೋತೀನಿ" is a
    correct rendering of what was said, and 0.9 flagged it as an error.

    Note this cannot separate Hindi from Marathi - both are Devanagari, so
    SCRIPT_RANGES gives them the identical range and the guard is a no-op
    between those two. Only Whisper's own acoustic detection distinguishes
    them.
    """
    text = run_whisper(audio, lang)

    if lang == "en" or not text or script_ok(text, lang, ASR_SCRIPT_THRESHOLD):
        return text, None

    print(f"Wrong script for {lang}: {text[:60]!r} -> retrying with script prompt")
    retry = run_whisper(audio, lang, WHISPER_PROMPTS.get(lang))

    if retry and script_ok(retry, lang, ASR_SCRIPT_THRESHOLD):
        return retry, None

    return (retry or text), (
        f"I am not sure I understood this clearly in {LANGUAGE_NAMES.get(lang, lang)}. "
        "Please check the text below, correct it, or record again closer to the microphone."
    )


# ============================================================
# WHISPER VOICE TRANSCRIPTION
# ============================================================

def _fail(message: str, status: int = 400, **extra):
    """Error replies now carry a real HTTP status.

    Every failure used to come back as 200 with {"success": false}, so any
    client using raise_for_status() - including ai/testai.py - read a dead
    request as a successful one.
    """
    body = {"success": False, "text": "", "error": message}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(""),      # the language selected in the app (hint)
    force: str = Form("0"),        # "1" = trust the app language for transcription
):
    temp_file_path = None

    try:
        if whisper_model is None:
            return _fail("Whisper model is not loaded. Check the backend log.", 503)

        if not FFMPEG_PATH:
            return _fail(
                "ffmpeg is not installed or not on PATH, so the recording cannot "
                "be decoded. Install ffmpeg and restart the backend.", 503)

        hint = normalize_language(language, default=None) if language else None

        audio_bytes = file.file.read()
        if not audio_bytes:
            return _fail("No audio data received.", 400)

        # Keep the client's real extension. Forcing ".wav" onto WebM/Opus bytes
        # left ffmpeg to sniff the container past a lying filename.
        suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(audio_bytes)
        tmp.close()
        temp_file_path = tmp.name

        t0 = time.time()

        # Decode ONCE and reuse the waveform for detection and transcription.
        try:
            audio = whisper.load_audio(temp_file_path)
        except Exception as e:
            return _fail(f"Could not decode the audio ({e}). Please record again.", 400)

        # Whisper invents fluent sentences out of silence, so refuse the clip
        # rather than hand the chat model a hallucination.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < SILENCE_PEAK:
            return _fail(
                "No speech was detected in that recording. Please record again and "
                "speak closer to the microphone.", 400,
                peak=round(peak, 5), language=hint)

        # 1) which language was actually spoken?
        chosen, detected, probs, low_confidence = detect_spoken_language(audio, hint)

        # "Force" decides which language we TRANSCRIBE in. It must not overwrite
        # what Whisper actually heard: the old code set detected = hint too, so
        # hint_mismatch could never fire under force and the UI was handed a
        # fabricated {hint: 1.0} that it displayed as "100% confidence".
        forced = force == "1" and bool(hint)
        if forced:
            chosen = hint

        supported_mass = sum(probs.values())
        top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

        print("==========================================")
        print("VOICE REQUEST")
        print("Model          :", WHISPER_MODEL_NAME)
        print("Audio size     :", len(audio_bytes), "bytes |", suffix)
        print("Peak amplitude :", round(peak, 4))
        print("App hint       :", hint, "| forced:", forced)
        print("Whisper top-3  :", [(c, round(p, 3)) for c, p in top3])
        print("Supported mass :", round(supported_mass, 3))
        print("Whisper pick   :", detected, "| low confidence:", low_confidence)
        print("Language used  :", chosen)
        print(f"Detection took : {time.time() - t0:.1f}s")
        print("==========================================")

        # 2) transcribe in that language (with wrong-script guard)
        text, warning = transcribe_with_guard(audio, chosen)

        print("Text:", text)
        print(f"Total /transcribe time: {time.time() - t0:.1f}s")

        if not text:
            return _fail("Could not understand the audio.", 400, language=chosen)

        if low_confidence and not warning:
            warning = (
                "I could not confidently tell which language that was. "
                f"I transcribed it as {LANGUAGE_NAMES.get(chosen, chosen)} - "
                "please check the text below before sending it."
            )

        return {
            "success": True,
            "text": text,
            "language": chosen,               # code used for transcription + reply
            "detected_language": detected,    # raw Whisper guess (for debugging)
            "forced": forced,
            "low_confidence": low_confidence,
            # Raw Whisper probabilities restricted to our six languages. These
            # do NOT sum to 1 - supported_mass is how much of Whisper's total
            # belief landed on a language Neravu supports at all. The UI used
            # to render them as percentages, implying they did.
            "probabilities": {c: round(p, 3) for c, p in top3},
            "supported_mass": round(supported_mass, 3),
            "warning": warning,
            # Compared against what Whisper HEARD, not against the language we
            # transcribed in - otherwise forcing always silenced this.
            "hint_mismatch": bool(
                hint and hint != detected and hint != "en" and detected != "en"
            ),
            "app_language": hint,
        }

    except Exception as e:
        print("WHISPER ERROR:", str(e))
        return _fail(str(e), 500)

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
