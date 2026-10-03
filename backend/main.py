# ============================================================
# NERAVU BACKEND
# FastAPI + Whisper + Ollama
# ============================================================

import os
import tempfile
import requests
import time
import torch
import whisper

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware


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

# "small" is weak for Kannada and often mistakes it for Hindi/Telugu.
# Use "medium" (good balance) or "large-v3" (best, needs ~10 GB RAM/VRAM).
# Override without editing code:  set WHISPER_MODEL=large-v3
# Default: "medium" only if a GPU is available, otherwise "small" (CPU-friendly).
HAS_GPU = torch.cuda.is_available()
# "small" writes Kannada as Devanagari gibberish. Use "medium" at minimum.
# If medium is too slow on your PC:  set WHISPER_MODEL=small  (accuracy drops a lot)
# Best accuracy (needs ~10 GB RAM/VRAM):  set WHISPER_MODEL=large-v3
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")

# If the user picked an Indian language in the app and Whisper's top guess is a
# DIFFERENT Indian language, we trust the user's choice as long as Whisper gave
# it at least this fraction of the top guess's probability.
# (Fixes Kannada being detected as Hindi/Telugu. English is never overridden.)
HINT_BIAS = 0.25


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


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "Neravu backend is running", "status": "success"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "whisper_model": WHISPER_MODEL_NAME,
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
    if code == "en":
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
    probs : {lang_code: probability} from Whisper (any codes)
    hint  : language code the user selected in the app (or None)

    Returns (chosen_code, detected_code, restricted_probs)

    - Only our 6 supported languages are considered.
    - Best Whisper guess wins, EXCEPT when both the guess and the user's
      hint are Indian languages and the hint is a plausible runner-up
      (>= HINT_BIAS * best). English is never overridden in either direction,
      so English speech with Kannada selected still comes out as English.
    """
    restricted = {c: float(probs.get(c, 0.0)) for c in SUPPORTED}
    detected = max(restricted, key=restricted.get)
    chosen = detected

    if (
        hint in restricted
        and hint != detected
        and hint != "en"
        and detected != "en"
        and restricted[hint] >= HINT_BIAS * restricted[detected]
    ):
        chosen = hint

    return chosen, detected, restricted


def detect_spoken_language(path: str, hint: str = None):
    audio = whisper.pad_or_trim(whisper.load_audio(path))
    n_mels = getattr(whisper_model.dims, "n_mels", 80)
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(whisper_model.device)
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


def run_whisper(path, lang, prompt=None):
    result = whisper_model.transcribe(
        path,
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


def transcribe_with_guard(path, lang):
    """
    Transcribe in `lang`. If the text comes out in the WRONG SCRIPT
    (e.g. Devanagari letters for Kannada speech), retry once with a short
    prompt written in the right script. Returns (text, warning_or_None).
    """
    text = run_whisper(path, lang)

    if lang == "en" or not text or script_ok(text, lang):
        return text, None

    print(f"Wrong script for {lang}: {text[:60]!r} -> retrying with script prompt")
    retry = run_whisper(path, lang, WHISPER_PROMPTS.get(lang))

    if retry and script_ok(retry, lang):
        return retry, None

    return (retry or text), (
        f"I am not sure I understood this clearly in {LANGUAGE_NAMES.get(lang, lang)}. "
        "Please check the text below, correct it, or record again closer to the microphone."
    )


# ============================================================
# WHISPER VOICE TRANSCRIPTION
# ============================================================

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(""),      # the language selected in the app (hint)
    force: str = Form("0"),        # "1" = trust the app language, skip auto-detection
):
    temp_file_path = None

    try:
        if whisper_model is None:
            return {"success": False, "text": "", "error": "Whisper model is not loaded."}

        hint = normalize_language(language, default=None) if language else None

        audio_bytes = file.file.read()
        if not audio_bytes:
            return {"success": False, "text": "", "error": "No audio data received."}

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        temp_file_path = tmp.name

        t0 = time.time()

        # 1) which language was actually spoken?
        if force == "1" and hint:
            chosen, detected, probs = hint, hint, {hint: 1.0}
            print("FORCED language from app:", hint)
        else:
            chosen, detected, probs = detect_spoken_language(temp_file_path, hint)

        top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("==========================================")
        print("VOICE REQUEST")
        print("Model          :", WHISPER_MODEL_NAME)
        print("Audio size     :", len(audio_bytes), "bytes")
        print("App hint       :", hint, "| forced:", force == "1")
        print("Whisper top-3  :", [(c, round(p, 3)) for c, p in top3])
        print("Whisper pick   :", detected)
        print("Language used  :", chosen)
        print(f"Detection took : {time.time() - t0:.1f}s")
        print("==========================================")

        # 2) transcribe in that language (with wrong-script guard)
        text, warning = transcribe_with_guard(temp_file_path, chosen)

        print("Text:", text)
        print(f"Total /transcribe time: {time.time() - t0:.1f}s")

        if not text:
            return {"success": False, "text": "", "language": chosen,
                    "error": "Could not understand the audio."}

        return {
            "success": True,
            "text": text,
            "language": chosen,               # code used for transcription + reply
            "detected_language": detected,    # raw Whisper guess (for debugging)
            "probabilities": {c: round(p, 3) for c, p in top3},
            "warning": warning,
            # heard a different Indian language than the one selected in the app
            "hint_mismatch": bool(hint and hint != chosen and hint != "en" and chosen != "en"),
            "app_language": hint,
        }

    except Exception as e:
        print("WHISPER ERROR:", str(e))
        return {"success": False, "text": "", "error": str(e)}

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass