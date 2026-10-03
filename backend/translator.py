import requests


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

LANGUAGE_CODES = {
    "Kannada": "kn",
    "Telugu": "te",
    "Tamil": "ta",
    "Hindi": "hi",
    "Marathi": "mr"
}


# ============================================================
# UNICODE SCRIPT RANGES
# ============================================================

SCRIPT_RANGES = {
    "Kannada": (0x0C80, 0x0CFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Hindi": (0x0900, 0x097F),
    "Marathi": (0x0900, 0x097F)
}


# ============================================================
# CHECK FOR WRONG SCRIPT
# ============================================================

def has_wrong_script(text, language):

    if language not in SCRIPT_RANGES:
        return False

    start, end = SCRIPT_RANGES[language]

    for char in text:

        if not char.isalpha():
            continue

        code = ord(char)

        if not (start <= code <= end):
            return True

    return False


# ============================================================
# CHECK FOR EXTRA MODEL TEXT
# ============================================================

def looks_like_prompt(text):

    bad_patterns = [
        "1.",
        "2.",
        "3.",
        "4.",
        "5.",
        "6.",
        "7.",
        "8.",
        "9.",
        "10.",
        "IMPORTANT",
        "Important:",
        "Translation:",
        "translation:",
        "English:",
        "English text:",
        "Kannada:",
        "Telugu:",
        "Tamil:",
        "Hindi:",
        "Marathi:"
    ]

    return any(
        pattern in text
        for pattern in bad_patterns
    )


# ============================================================
# TRANSLATION
# ============================================================

def translate_to_language(text, language):

    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    if language == "English":
        return text


    # --------------------------------------------------------
    # UNSUPPORTED LANGUAGE
    # --------------------------------------------------------

    if language not in LANGUAGE_CODES:
        return text


    # --------------------------------------------------------
    # LANGUAGE-SPECIFIC INSTRUCTIONS
    # --------------------------------------------------------

    language_instructions = {

        "Kannada": (
            "Translate ONLY into Kannada. "
            "Use Kannada script only. "
            "Do NOT use Telugu script. "
            "For fever, use the Kannada word 'ಜ್ವರ'. "
            "Do NOT use the Telugu word 'జ్వరం'."
        ),

        "Telugu": (
            "Translate ONLY into Telugu. "
            "Use Telugu script only. "
            "Do NOT use Kannada script."
        ),

        "Tamil": (
            "Translate ONLY into Tamil. "
            "Use Tamil script only. "
            "Do NOT use Kannada or Telugu script."
        ),

        "Hindi": (
            "Translate ONLY into Hindi. "
            "Use Devanagari script only. "
            "Do NOT use Kannada, Telugu, or Tamil script."
        ),

        "Marathi": (
            "Translate ONLY into Marathi. "
            "Use Devanagari script only. "
            "Do NOT use Kannada, Telugu, or Tamil script."
        )
    }


    instruction = language_instructions[language]


    # --------------------------------------------------------
    # MAIN TRANSLATION PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a translation system for an elderly healthcare assistant.

TARGET LANGUAGE:
{language}

{instruction}

STRICT RULES:

1. Translate ONLY the English sentence.
2. Output ONLY the translation.
3. Do NOT output English.
4. Do NOT output another Indian language.
5. Do NOT mix scripts.
6. Do NOT add explanations.
7. Do NOT repeat the instructions.
8. Do NOT add numbering.
9. Keep the meaning unchanged.
10. Keep the sentence simple and natural.

ENGLISH SENTENCE:
{text}

{language} TRANSLATION:
"""


    last_result = text


    # ========================================================
    # TRY TRANSLATION UP TO 3 TIMES
    # ========================================================

    for attempt in range(3):

        try:

            response = requests.post(

                "http://localhost:11434/api/chat",

                json={

                    "model": "translategemma:4b",

                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    "stream": False,

                    "options": {
                        "temperature": 0,
                        "num_predict": 100
                    }
                },

                timeout=180
            )


            response.raise_for_status()


            result = response.json()[
                "message"
            ][
                "content"
            ].strip()


            last_result = result


            print(
                f"\nTranslation attempt {attempt + 1}"
            )

            print(
                f"Target language: {language}"
            )

            print(
                f"Result: {result}"
            )


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            wrong_script = has_wrong_script(
                result,
                language
            )

            extra_text = looks_like_prompt(
                result
            )


            # ------------------------------------------------
            # VALID TRANSLATION
            # ------------------------------------------------

            if not wrong_script and not extra_text:

                return result


            # ------------------------------------------------
            # INVALID TRANSLATION
            # ------------------------------------------------

            print(
                "Translation validation failed."
            )

            print(
                "Retrying with stricter instructions..."
            )


            # ------------------------------------------------
            # STRICT RETRY PROMPT
            # ------------------------------------------------

            prompt = f"""
The previous translation was INVALID.

Translate this sentence into {language}.

TARGET LANGUAGE:
{language}

VERY IMPORTANT:

- Output ONLY the translation.
- Do NOT output English.
- Do NOT use another Indian language.
- Do NOT mix languages.
- Do NOT mix scripts.
- Do NOT add explanations.
- Do NOT add labels.
- Do NOT add numbering.

If the target language is Kannada:

Use Kannada script only.

Correct Kannada example:
ಜ್ವರ

Incorrect Telugu word:
జ్వరం

English sentence:
{text}

Return ONLY the {language} translation.
"""


        except Exception as e:

            print(
                f"Translation error: {e}"
            )


    # ========================================================
    # ALL ATTEMPTS FAILED
    # ========================================================

    print(
        f"Warning: Could not validate {language} translation."
    )

    return last_result