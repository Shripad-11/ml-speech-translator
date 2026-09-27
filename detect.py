from langdetect import detect

def fix_language(lang, text):
    if lang == "hi":
        if "आहे" in text or "माझं" in text:
            return "mr"
    return lang

def detect_language(text):
    lang = detect(text)
    lang = fix_language(lang, text)
    return lang