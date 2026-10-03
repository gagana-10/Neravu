import whisper
import os

# ============================================================
# LANGUAGE SETTINGS
# ============================================================

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr"
}

# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper model...")
model = whisper.load_model("base")

print("\nWhisper is ready!")

# ============================================================
# SELECT LANGUAGE
# ============================================================

print("\nAvailable languages:")

for number, language in enumerate(LANGUAGES.keys(), start=1):
    print(f"{number}. {language}")

choice = input("\nEnter language number: ")

try:
    choice = int(choice)

    language_names = list(LANGUAGES.keys())

    if choice < 1 or choice > len(language_names):
        print("Invalid language choice.")
        exit()

    selected_language = language_names[choice - 1]
    language_code = LANGUAGES[selected_language]

except ValueError:
    print("Please enter a number.")
    exit()

# ============================================================
# AUDIO FILE
# ============================================================

audio_file = "audio.wav"

if not os.path.exists(audio_file):
    print("\nERROR: audio.wav not found!")

    print("\nPlace your audio file here:")
    print(os.path.abspath(audio_file))

    exit()

# ============================================================
# TRANSCRIBE
# ============================================================

print("\n========================================")
print("SELECTED LANGUAGE")
print("========================================")
print(selected_language)

print("\nTranscribing...")
print("Please wait...\n")

result = model.transcribe(
    audio_file,
    language=language_code,
    task="transcribe",
    fp16=False
)

# ============================================================
# RESULT
# ============================================================

text = result["text"].strip()

print("========================================")
print("WHISPER TRANSCRIPTION")
print("========================================")
print(text)
print("========================================")

print("\nLanguage:", selected_language)
print("Language code:", language_code)

