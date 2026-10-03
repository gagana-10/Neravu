"""
Checks for the pure decision logic behind voice transcription.

Runs WITHOUT torch, whisper or ffmpeg: it parses main.py and executes only
the two functions under test, so it is usable on any machine.

    python test_whisper_logic.py
"""

import ast
import io
import os
import sys


# ============================================================
# LOAD THE FUNCTIONS UNDER TEST
# ============================================================

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

FUNCTIONS = {"choose_language", "script_ok"}
CONSTANTS = {"SUPPORTED", "HINT_BIAS", "MIN_LANG_CONFIDENCE",
             "SCRIPT_RANGES", "ASR_SCRIPT_THRESHOLD"}


def load_logic():
    source = io.open(MAIN, encoding="utf-8").read()
    tree = ast.parse(source)

    pieces = []

    for node in tree.body:

        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            pieces.append(ast.get_source_segment(source, node))

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in CONSTANTS:
                    pieces.append(ast.get_source_segment(source, node))

    namespace = {}
    exec("\n\n".join(pieces), namespace)
    return namespace


LOGIC = load_logic()

choose_language = LOGIC["choose_language"]
script_ok = LOGIC["script_ok"]

MIN_LANG_CONFIDENCE = LOGIC["MIN_LANG_CONFIDENCE"]
ASR_SCRIPT_THRESHOLD = LOGIC["ASR_SCRIPT_THRESHOLD"]


# ============================================================
# TEST HARNESS
# ============================================================

failures = []


def check(label, got, want):
    passed = got == want

    print(f"{'PASS' if passed else 'FAIL'}  {label}")

    if not passed:
        print(f"        got  {got!r}")
        print(f"        want {want!r}")
        failures.append(label)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SAMPLE TEXT
# ============================================================

KANNADA_PURE = "ನನಗೆ ಜ್ವರ ಇದೆ ಮತ್ತು ತಲೆನೋವು"
KANNADA_MIXED = "ಸಕ್ಕರೆ ಕಾಯಿಲೆ ರತು, sugar tablet ತಗೋತೀನಿ"
DEVANAGARI = "मुझे बुखार है और सिर दर्द है"


# ============================================================
# LANGUAGE CHOICE
# ============================================================

section("choose_language")

check(
    "confident Kannada with no hint is used as-is",
    choose_language({"kn": 0.81, "hi": 0.05, "en": 0.02})[:2],
    ("kn", "kn")
)

check(
    "a plausible runner-up hint still wins (HINT_BIAS)",
    choose_language({"hi": 0.50, "kn": 0.30}, "kn")[:2],
    ("kn", "hi")
)

check(
    "English is never overridden by an Indian hint",
    choose_language({"en": 0.70, "kn": 0.20}, "kn")[:2],
    ("en", "en")
)

# Silence and unsupported languages leave all six of our languages near zero.
# HINT_BIAS is a RELATIVE test, so without a floor it used to promote a hint
# sitting at a probability of 0.002 and report it as a real detection.
chosen, detected, probs, low = choose_language(
    {"kn": 0.004, "hi": 0.002, "ja": 0.6}, "kn"
)

check("silence / unsupported speech is flagged low-confidence", low, True)
check("  ...and falls back to the language the user chose", chosen, "kn")

check(
    "low confidence with no hint is still flagged",
    choose_language({"kn": 0.004, "hi": 0.002, "ja": 0.6})[3],
    True
)

check(
    f"a probability of exactly {MIN_LANG_CONFIDENCE} is not low-confidence",
    choose_language({"kn": MIN_LANG_CONFIDENCE})[3],
    False
)

check(
    f"a probability just under {MIN_LANG_CONFIDENCE} is low-confidence",
    choose_language({"kn": MIN_LANG_CONFIDENCE - 0.001})[3],
    True
)


# ============================================================
# SCRIPT GUARD
# ============================================================

section(f"script_ok   (speech {ASR_SCRIPT_THRESHOLD}, generated text 0.9)")

check(
    "code-switched Kannada was rejected at the old 0.9 threshold",
    script_ok(KANNADA_MIXED, "kn", 0.9),
    False
)

check(
    "code-switched Kannada is accepted at the speech threshold",
    script_ok(KANNADA_MIXED, "kn", ASR_SCRIPT_THRESHOLD),
    True
)

check(
    "pure Kannada passes both thresholds",
    (script_ok(KANNADA_PURE, "kn", 0.9),
     script_ok(KANNADA_PURE, "kn", ASR_SCRIPT_THRESHOLD)),
    (True, True)
)

check(
    "Devanagari written for Kannada speech is still caught",
    script_ok(DEVANAGARI, "kn", ASR_SCRIPT_THRESHOLD),
    False
)

check(
    "Latin transliteration of Kannada is still caught",
    script_ok("nanage jvara ide mattu tale novu", "kn", ASR_SCRIPT_THRESHOLD),
    False
)

check(
    "an unknown language code does not raise KeyError",
    script_ok("hello there", "zz", ASR_SCRIPT_THRESHOLD),
    True
)

check("empty text is not acceptable", script_ok("", "kn"), False)
check("digits-only text is not acceptable", script_ok("123 456", "kn"), False)

# Hindi and Marathi are both Devanagari, so SCRIPT_RANGES gives them the same
# range and this guard cannot tell them apart. Only Whisper's own acoustic
# detection can. Asserted so the limitation is visible rather than assumed.
check(
    "the script guard cannot separate Hindi from Marathi",
    (script_ok(DEVANAGARI, "hi"), script_ok(DEVANAGARI, "mr")),
    (True, True)
)


# ============================================================
# RESULT
# ============================================================

section("RESULT")

if failures:
    print(f"{len(failures)} check(s) failed:")
    for name in failures:
        print("  -", name)
    sys.exit(1)

print("All checks passed.")
