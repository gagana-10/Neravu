from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import uuid
import traceback

# ------------------------------------------------------------------
# NOTE: this is a STANDALONE scratch server for testing Whisper on its
# own. It is NOT what the Streamlit app talks to -- that is the
# /transcribe endpoint in backend/main.py, which also does language
# detection and the wrong-script guard.
#
# It used to listen on port 8000, the same port uvicorn serves the real
# backend on, while expecting the upload field "audio" where the real
# backend expects "file". Starting this instead of the backend therefore
# answered the app's requests with
#     400 "No audio file received. Expected field name: audio"
# which looks exactly like a broken frontend. It now uses port 8001 and
# accepts either field name.
# ------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

PORT = 8001
UPLOAD_FIELDS = ("file", "audio")

print("Loading Whisper model...")

# Use tiny first for reliable CPU testing.
# Once everything works, we can change this to "base".
model = whisper.load_model("tiny")

print("Whisper model loaded successfully!")


@app.route("/transcribe", methods=["POST"])
def transcribe():

    print("\n========== TRANSCRIPTION REQUEST ==========")

    try:
        # --------------------------------------------------
        # 1. Check whether a file was actually received
        # --------------------------------------------------

        print("Content-Type:", request.content_type)
        print("Received files:", list(request.files.keys()))

        field = next((f for f in UPLOAD_FIELDS if f in request.files), None)

        if field is None:
            return jsonify({
                "success": False,
                "error": "No audio file received. Expected field name: "
                         + " or ".join(UPLOAD_FIELDS)
            }), 400

        audio_file = request.files[field]

        if audio_file.filename == "":
            return jsonify({
                "success": False,
                "error": "Audio filename is empty."
            }), 400

        print("Received filename:", audio_file.filename)

        # --------------------------------------------------
        # 2. Save uploaded audio
        # --------------------------------------------------

        temp_dir = os.path.join(os.path.dirname(__file__), "temp_audio")

        os.makedirs(temp_dir, exist_ok=True)

        # keep the real extension instead of claiming every upload is a WAV
        extension = os.path.splitext(audio_file.filename)[1].lower() or ".wav"

        filename = f"{uuid.uuid4().hex}{extension}"

        audio_path = os.path.join(temp_dir, filename)

        audio_file.save(audio_path)

        print("Saved audio:", audio_path)

        # --------------------------------------------------
        # 3. Check file exists and has data
        # --------------------------------------------------

        if not os.path.exists(audio_path):
            return jsonify({
                "success": False,
                "error": "Audio file could not be saved."
            }), 500

        file_size = os.path.getsize(audio_path)

        print("Audio file size:", file_size, "bytes")

        if file_size == 0:
            os.remove(audio_path)

            return jsonify({
                "success": False,
                "error": "Received audio file is empty."
            }), 400

        # --------------------------------------------------
        # 4. Transcribe
        # --------------------------------------------------

        print("Starting Whisper transcription...")

        result = model.transcribe(
            audio_path,
            task="transcribe",
            fp16=False,
            verbose=False
        )

        text = result.get("text", "").strip()

        detected_language = result.get("language", "unknown")

        print("Detected language:", detected_language)
        print("Transcription:", text)

        # --------------------------------------------------
        # 5. Delete temporary file
        # --------------------------------------------------

        try:
            os.remove(audio_path)
        except Exception:
            pass

        # --------------------------------------------------
        # 6. Return result
        # --------------------------------------------------

        return jsonify({
            "success": True,
            "text": text,
            "language": detected_language
        }), 200

    except Exception as e:

        print("\n========== WHISPER ERROR ==========")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "Whisper API is running",
        "endpoint": "/transcribe",
        "method": "POST",
        "field": " or ".join(UPLOAD_FIELDS),
        "port": PORT,
        "note": "Scratch test server. The app uses backend/main.py on port 8000."
    })


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=True
    )