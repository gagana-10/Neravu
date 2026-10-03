from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import uuid
import traceback

app = Flask(__name__)
CORS(app)

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

        if "audio" not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio file received. Expected field name: audio"
            }), 400

        audio_file = request.files["audio"]

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

        filename = f"{uuid.uuid4().hex}.wav"

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
        "field": "audio"
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )