# Multilingual Speech-to-Speech Translation Pipeline

An end-to-end AI application built with Flask that transcribes input speech, translates the text into target regional languages using Meta's NLLB model, and synthesizes the translated output back into spoken audio.

##  Key Features

* **Speech-to-Text (STT):** Uses `faster-whisper` (`small` model) with Voice Activity Detection (VAD) for noise-resilient transcription.
* **Translation Engine:** Meta's `facebook/nllb-200-distilled-600M` model optimized for regional Indian language translation accuracy and inference speed.
* **Text-to-Speech (TTS):** Converts translated output back to spoken audio using `gTTS`.
* **Hardware Auto-Detection:** Dynamically switches execution across NVIDIA GPUs (`cuda`), Apple Silicon (`mps`), or CPU (`qint8` quantized).
* **Text Pre/Post-Processing:** Sentence-level segmentation using regex to preserve long-context meaning and custom post-processing rules for technical domain terminology.

##  Tech Stack

* **Backend:** Python, Flask
* **Machine Learning & NLP:** PyTorch, Hugging Face Transformers, Faster-Whisper, gTTS
* **Supported Languages:** English (en), Hindi (hi), Marathi (mr), Punjabi (pa), Telugu (te), Gujarati (gu)

