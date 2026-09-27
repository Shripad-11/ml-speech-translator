from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import re

# Cache to store the loaded model and tokenizer in memory
cache = {}

# NLLB language code mapping for your supported languages
NLLB_LANG_CODES = {
    "en": "eng_Latn", 
    "hi": "hin_Deva", 
    "mr": "mar_Deva",
    "pa": "pan_Guru", 
    "te": "tel_Telu", 
    "gu": "guj_Gujr",
}

# The 600M model: Fast, lightweight, and perfect for GitHub distribution
MODEL_NAME = "facebook/nllb-200-distilled-600M"

def load_nllb_model():
    """Loads the NLLB model with hardware acceleration and optimization."""
    if MODEL_NAME not in cache:
        print(f"⏳ Loading Optimized NLLB ({MODEL_NAME})...")
        
        # Load tokenizer and suppress the deprecation warning
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            use_fast=True, 
            clean_up_tokenization_spaces=True
        )
        
        # 🏎️ Hardware Detection (NVIDIA -> MacBook MPS -> CPU)
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16 
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        # Load the model with memory mapping
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=dtype, 
            low_cpu_mem_usage=True
        ).to(device)

        # ⚡ CPU Optimization: Quantize math to 8-bit to double inference speed
        if device == "cpu":
            print("⚙️ Applying Dynamic Quantization for CPU...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        cache[MODEL_NAME] = {"tokenizer": tokenizer, "model": model, "device": device}
        print(f"✅ Model successfully loaded on {device.upper()}")
        
    return cache[MODEL_NAME]["tokenizer"], cache[MODEL_NAME]["model"], cache[MODEL_NAME]["device"]

def post_process(text):
    """Clean-up layer for technical terms and common AI slips."""
    corrections = {
        "extc": "EXTC",
        "trish": "Trish",
        "ai based": "AI-based",
        "mumbai": "Mumbai",
        "vidyalankar": "Vidyalankar"
    }
    for wrong, right in corrections.items():
        # Case-insensitive replacement to catch variations
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(right, text)
    return text

def translate(text, src, tgt):
    """Translates text from source language to target language using NLLB."""
    # If same language, skip translation
    if src == tgt: 
        return text

    src_nllb = NLLB_LANG_CODES.get(src)
    tgt_nllb = NLLB_LANG_CODES.get(tgt)

    if not src_nllb or not tgt_nllb:
        return f"Error: Unsupported language code ({src} or {tgt})"

    try:
        tokenizer, model, device = load_nllb_model()
        
        # Explicitly set the source language for the tokenizer
        tokenizer.src_lang = src_nllb

        # 1. SEGMENTATION: Split into sentences so the AI doesn't lose context
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
        if not sentences: 
            return ""

        # 2. BATCH TOKENIZATION: Process all sentences simultaneously
        inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)

        # Safely get the target language ID token
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

        # 3. HIGH-ACCURACY GENERATION
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=forced_bos_token_id, 
                max_new_tokens=128,
                num_beams=4,            # Beam search for robust grammar
                repetition_penalty=1.3, # Prevents looping/stuttering
                no_repeat_ngram_size=3
            )

        # 4. DECODE AND CLEAN
        translated_parts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        final_output = " ".join(translated_parts)
        
        return post_process(final_output)

    except Exception as e:
        print(f"❌ TRANSLATION ERROR: {str(e)}")
        return f"Translation error: {str(e)}"