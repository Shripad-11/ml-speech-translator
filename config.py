from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "static/generated"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "dataset_raw"
MODEL_DIR = BASE_DIR / "artifacts"

# Updated to reflect our new universal NLLB model instead of Helsinki-NLP
BASE_MODEL = "facebook/nllb-200-distilled-600M"

LANGS = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "pa": "Punjabi",
    "te": "Telugu",
    "gu": "Gujarati"
}