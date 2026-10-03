"""
Drives the real /transcribe endpoint over HTTP with a stubbed Whisper.

Whisper and torch are replaced with fakes, so this runs WITHOUT the 2.5 GB
torch download, without a Whisper model and without ffmpeg. What it checks is
Neravu's own decision logic -- language choice, the silence guard, the
wrong-script guard, the forced-recognition path and the HTTP contract -- not
Whisper's transcription accuracy, which only a real model can show.

    pip install fastapi python-multipart httpx numpy requests
    python test_transcribe_endpoint.py
"""

import os
import sys
import types

import numpy as np


# ============================================================
# FAKE TORCH
# ============================================================

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = fake_torch


# ============================================================
# FAKE WHISPER
# ============================================================

class FakeWhisperModel:

    def __init__(self):
        self.dims = types.SimpleNamespace(n_mels=80)
        self.device = "cpu"

        # set by each test
        self.language_probs = {}
        self.transcripts = []

        # observed by each test
        self.transcribe_calls = []

    def detect_language(self, mel):
        return None, dict(self.language_probs)

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls.append(kwargs)

        if self.transcripts:
            return {"text": self.transcripts.pop(0)}

        return {"text": ""}


MODEL = FakeWhisperModel()

STATE = {"audio": np.zeros(16000, dtype=np.float32), "load_audio_calls": 0}


def fake_load_audio(path):
    STATE["load_audio_calls"] += 1
    return STATE["audio"]


fake_whisper = types.ModuleType("whisper")
fake_whisper.load_model = lambda name: MODEL
fake_whisper.load_audio = fake_load_audio
fake_whisper.pad_or_trim = lambda audio, **kw: audio
fake_whisper.log_mel_spectrogram = lambda audio, **kw: types.SimpleNamespace(
    to=lambda device: None
)
sys.modules["whisper"] = fake_whisper


# ============================================================
# IMPORT THE REAL BACKEND
# ============================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main                                    # noqa: E402
from fastapi.testclient import TestClient      # noqa: E402

client = TestClient(main.app, raise_server_exceptions=False)


# ============================================================
# SAMPLE AUDIO AND TEXT
# ============================================================

SPEECH = (np.sin(np.linspace(0, 400, 16000)) * 0.5).astype(np.float32)
SILENCE = np.full(16000, 0.0005, dtype=np.float32)

KANNADA_PURE = "ನನಗೆ ಜ್ವರ ಇದೆ ಮತ್ತು ತಲೆನೋವು"
KANNADA_MIXED = "ಸಕ್ಕರೆ ಕಾಯಿಲೆ ರತು, sugar tablet ತಗೋತೀನಿ"
DEVANAGARI = "मुझे बुखार है और सिर दर्द है"

failures = []


# ============================================================
# HELPERS
# ============================================================

def arrange(probs, transcripts, audio=SPEECH, ffmpeg="/usr/bin/ffmpeg"):
    MODEL.language_probs = probs
    MODEL.transcripts = list(transcripts)
    MODEL.transcribe_calls = []
    STATE["audio"] = audio
    STATE["load_audio_calls"] = 0
    main.FFMPEG_PATH = ffmpeg


def post(language="", force="0", payload=b"RIFFfake-audio-bytes"):
    return client.post(
        "/transcribe",
        files={"file": ("voice.wav", payload, "audio/wav")},
        data={"language": language, "force": force},
    )


def check(label, got, want):
    passed = got == want
    print(("PASS  " if passed else "FAIL  ") + label)
    if not passed:
        print("        got  " + repr(got))
        print("        want " + repr(want))
        failures.append(label)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SYMPTOM 1 - THE REQUEST FAILS
# ============================================================

section("SYMPTOM 1: the request fails")

arrange({"kn": 0.9}, [KANNADA_PURE], ffmpeg=None)
response = post(language="Kannada")

check("missing ffmpeg returns 503, not a swallowed error",
      response.status_code, 503)
check("  ...and the reason names ffmpeg",
      "ffmpeg" in response.json()["error"], True)

arrange({"kn": 0.9}, [KANNADA_PURE])
check("an empty upload returns 400", post(language="Kannada", payload=b"").status_code, 400)

arrange({"kn": 0.9}, [""])
response = post(language="Kannada")
check("an unintelligible clip returns 400, not 200", response.status_code, 400)
check("  ...and reports failure", response.json()["success"], False)

arrange({"kn": 0.9}, [KANNADA_PURE])
response = post(language="Kannada")
check("a good clip still returns 200", response.status_code, 200)

# The old endpoint decoded the clip once for detection and again for every
# transcribe pass, so ffmpeg ran two or three times per request.
check("the audio is decoded exactly once per request",
      STATE["load_audio_calls"], 1)


# ============================================================
# SYMPTOM 2 - THE LANGUAGE IS WRONG
# ============================================================

section("SYMPTOM 2: the detected language is wrong")

# Whisper really heard Hindi; the user forced Kannada.
arrange({"hi": 0.72, "kn": 0.06}, [KANNADA_PURE])
body = post(language="Kannada", force="1").json()

check("forcing transcribes in the language the user chose",
      body["language"], "kn")
check("forcing does NOT overwrite what Whisper heard",
      body["detected_language"], "hi")
check("a mis-heard forced clip now warns the user",
      body["hint_mismatch"], True)
check("the forced flag reaches the UI", body["forced"], True)

# The old code returned {hint: 1.0} here, which the UI printed as "100%".
check("confidence figures are real, not fabricated",
      body["probabilities"].get("kn"), 0.06)
check("  ...so the fabricated 1.0 is gone",
      body["probabilities"].get("kn") == 1.0, False)

# Honest reporting of how much belief landed outside Neravu's six languages.
check("supported_mass exposes belief that fell outside our languages",
      body["supported_mass"], 0.78)

# Unforced: the hint is a plausible runner-up, so it still wins (HINT_BIAS).
arrange({"hi": 0.50, "kn": 0.30}, [KANNADA_PURE])
body = post(language="Kannada").json()
check("unforced, a plausible hint still overrides the guess",
      (body["language"], body["detected_language"]), ("kn", "hi"))

# English must never be overridden by an Indian hint.
arrange({"en": 0.80, "kn": 0.10}, ["I have a fever and a headache"])
check("English speech is not forced into Kannada",
      post(language="Kannada").json()["language"], "en")


# ============================================================
# SYMPTOM 3 - THE TRANSCRIBED TEXT IS WRONG
# ============================================================

section("SYMPTOM 3: the transcribed text is wrong")

# Silence used to become a confident, hallucinated sentence.
arrange({"kn": 0.9}, ["ನಿಮ್ಮ ಆರೋಗ್ಯ ಹೇಗಿದೆ"], audio=SILENCE)
response = post(language="Kannada")
check("a silent clip is refused instead of hallucinated",
      response.status_code, 400)
check("  ...with an explanation",
      "No speech" in response.json()["error"], True)
check("  ...and Whisper is never even asked",
      len(MODEL.transcribe_calls), 0)

# Noise: every supported language sits near zero.
arrange({"kn": 0.004, "hi": 0.002, "ja": 0.95}, [KANNADA_PURE])
body = post(language="Kannada").json()
check("an unrecognisable language is flagged low-confidence",
      body["low_confidence"], True)
check("  ...and the user is warned", bool(body["warning"]), True)

# Code-switched speech is CORRECT and must not be re-run or flagged.
arrange({"kn": 0.9}, [KANNADA_MIXED])
body = post(language="Kannada").json()
check("code-switched Kannada is accepted as-is", body["text"], KANNADA_MIXED)
check("  ...with no false 'I misunderstood' warning", body["warning"], None)
check("  ...and no wasted retry pass", len(MODEL.transcribe_calls), 1)

# A genuinely wrong script still triggers the retry.
arrange({"kn": 0.9}, [DEVANAGARI, KANNADA_PURE])
body = post(language="Kannada").json()
check("Devanagari written for Kannada triggers a retry",
      len(MODEL.transcribe_calls), 2)
check("  ...and the corrected text is returned", body["text"], KANNADA_PURE)
check("  ...with no warning once the retry succeeds", body["warning"], None)
check("  ...and the retry is steered with a Kannada prompt",
      MODEL.transcribe_calls[1]["initial_prompt"], main.WHISPER_PROMPTS["kn"])

# Retry fails too -> return the best text available, but say so.
arrange({"kn": 0.9}, [DEVANAGARI, DEVANAGARI])
check("a failed retry still warns the user",
      bool(post(language="Kannada").json()["warning"]), True)


# ============================================================
# HEALTH REPORTING
# ============================================================

section("Health reporting")

main.FFMPEG_PATH = None
body = client.get("/health").json()
check("health is 'degraded' without ffmpeg", body["status"], "degraded")
check("  ...and says voice is not ready", body["voice_ready"], False)

main.FFMPEG_PATH = "/usr/bin/ffmpeg"
check("health is 'ok' once ffmpeg is present",
      client.get("/health").json()["status"], "ok")


# ============================================================
# RESULT
# ============================================================

section("RESULT")

if failures:
    print(str(len(failures)) + " check(s) failed:")
    for name in failures:
        print("  - " + name)
    sys.exit(1)

print("All checks passed.")
