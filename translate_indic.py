import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import re

# Cache to store loaded models to save memory
cache = {}

LANG_CODES = {
    "en": "eng_Latn", 
    "hi": "hin_Deva", 
    "mr": "mar_Deva",
    "pa": "pan_Guru", 
    "te": "tel_Telu", 
    "gu": "guj_Gujr",
}

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

def get_model_name(src, tgt):
    if src == "en" and tgt != "en":
        return "ai4bharat/indictrans2-en-indic-dist-200M"
    elif src != "en" and tgt == "en":
        return "ai4bharat/indictrans2-indic-en-dist-200M"
    else:
        # Indic to Indic
        return "ai4bharat/indictrans2-indic-indic-dist-320M"

def load_indictrans2(model_name):
    if model_name not in cache:
        print(f"⏳ Loading IndicTrans2 ({model_name})...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
        
        if device == "cpu":
            print("⚙️ Applying Dynamic Quantization for CPU...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        ip = IndicProcessor(inference=True)

        cache[model_name] = {"tokenizer": tokenizer, "model": model, "device": device, "ip": ip}
        print(f"✅ Model successfully loaded on {device.upper()}")
        
    return cache[model_name]

def translate(text, src, tgt):
    if src == tgt: 
        return text

    src_code = LANG_CODES.get(src)
    tgt_code = LANG_CODES.get(tgt)

    if not src_code or not tgt_code:
        return f"Error: Unsupported language code ({src} or {tgt})"

    try:
        model_name = get_model_name(src, tgt)
        components = load_indictrans2(model_name)
        
        tokenizer = components["tokenizer"]
        model = components["model"]
        device = components["device"]
        ip = components["ip"]

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
        if not sentences:
            sentences = [text.strip()]

        # Preprocess using IndicProcessor
        batch = ip.preprocess_batch(sentences, src_lang=src_code, tgt_lang=tgt_code)

        inputs = tokenizer(batch, padding="longest", truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs, 
                use_cache=True, 
                min_length=0, 
                max_length=256, 
                num_beams=4
            )

        # Decode output
        decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Postprocess using IndicProcessor
        final_sentences = ip.postprocess_batch(decoded_output, lang=tgt_code)

        final_output = " ".join(final_sentences)
        return post_process(final_output)

    except Exception as e:
        print(f"❌ TRANSLATION ERROR: {str(e)}")
        return f"Translation error: {str(e)}"
