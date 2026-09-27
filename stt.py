from faster_whisper import WhisperModel
import noisereduce as nr
import librosa
import soundfile as sf
import os
import uuid

# UPGRADE: 'medium' is significantly more accurate for Indian languages.
# It handles Hindi, Marathi, Punjabi, Telugu, and Gujarati much better than 'small'.
# Using 'auto' compute_type preserves accuracy (int8 degrades Indic script quality).
model = WhisperModel("medium", device="cpu", compute_type="auto") 



def preprocess_audio(audio_path):
    print("🧹 Pre-processing audio (Noise Reduction)...")
    # Load audio with librosa (sr=16000 is Whisper's native sample rate)
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Apply noise reduction (prop_decrease controls how much noise to remove, 0.8 is usually safe)
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    
    # Save to a temporary clean wav file
    temp_path = f"temp_clean_{uuid.uuid4().hex}.wav"
    sf.write(temp_path, reduced_noise, sr)
    
    return temp_path

def speech_to_text(audio_path, language=None):
    print(f"⏳ Transcribing audio (Forced Language: {language if language else 'Auto'})...")
    
    # 1. Pre-process audio to remove background noise
    clean_audio_path = preprocess_audio(audio_path)
    
    try:
        # A single pass without locking the language is better for auto-detect.
        # But if the user explicitly provided a language, force it.
        kwargs = {
            "task": "transcribe",
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": dict(min_silence_duration_ms=300)
        }
        if language:
            kwargs["language"] = language

        segments, info = model.transcribe(clean_audio_path, **kwargs)

        text = ""
        for seg in segments:
            text += seg.text

        detected_lang = info.language
        confidence = info.language_probability
        
        print(f"✅ Transcription complete: {text[:40]}...")
        print(f"🌍 Final Language: {detected_lang} (Confidence: {confidence:.2f})")
        
    finally:
        # Always clean up the temporary processed audio file
        if os.path.exists(clean_audio_path):
            os.remove(clean_audio_path)
    
    return text.strip(), detected_lang