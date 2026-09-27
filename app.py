from flask import Flask, render_template, request
import uuid
import warnings

# Suppress the harmless multiprocessing resource_tracker warnings on shutdown/reload
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

from config import UPLOAD_DIR, GENERATED_DIR, LANGS
from stt import speech_to_text
from translate import translate
from tts import text_to_speech
from detect import detect_language

app = Flask(__name__)

#create folders if not exist
UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

#supported languages
VALID_LANGS = ["en", "hi", "mr", "pa", "te", "gu"]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            file = request.files["audio"]
            tgt = request.form["lang"]
            forced_src_lang = request.form.get("src_lang", "auto")

            #save uploaded file
            filename = f"{uuid.uuid4()}_{file.filename}"
            path = UPLOAD_DIR / filename
            file.save(path)

            # Step 1: Speech to Text
            # Pass the forced_src_lang to Whisper. If it's "auto", Whisper will auto-detect.
            text, whisper_lang = speech_to_text(str(path), language=forced_src_lang if forced_src_lang != "auto" else None)

            # Handle empty audio
            if not text.strip():
                return render_template(
                    "index.html",
                    result={"error": "No speech detected"},
                    langs=LANGS
                )

            # Step 2: Custom Language Detection
            detected_lang = detect_language(text)

            # Smart selection: Trust Whisper's audio-based language detection primarily.
            # langdetect (text-based) is unreliable for short sentences and often defaults to Hindi.
            src = whisper_lang

            # Ensure valid language
            if src not in VALID_LANGS:
                src = "hi"   # fallback default

            if tgt not in VALID_LANGS:
                tgt = "en"

            # Debug (helps you during testing)
            print("Detected Source:", src)
            print("Target Language:", tgt)
            print("Text:", text)

            # same language Checking
            if src == tgt:
                return render_template(
                    "index.html",
                    result={"error": "The detected language is the same as your target language! Please select a different language to translate."},
                    langs=LANGS
                )

            # Step 3: Translation (with pivot support)
            translated = translate(text, src, tgt)

            #Step 4: Text to Speech
            out_file = GENERATED_DIR / f"{uuid.uuid4()}.mp3"
            text_to_speech(translated, tgt, str(out_file))

            # Result for UI
            result = {
                "original": text,
                "detected_lang": src,
                "target_lang": tgt,
                "translated": translated,
                "audio": out_file.name
            }

        except Exception as e:
            print("ERROR:", str(e))
            result = {"error": str(e)}

    return render_template("index.html", result=result, langs=LANGS)


if __name__ == "__main__":
    app.run(debug=True)