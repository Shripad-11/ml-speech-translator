from gtts import gTTS

def text_to_speech(text, lang, path):
    """
    Converts text to speech and saves it to the specified path.
    """
    # Simply pass the 'lang' code directly (e.g., 'en', 'hi', 'mr')
    tts = gTTS(text=text, lang=lang)
    tts.save(path)