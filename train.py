import os


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from config import BASE_MODEL, DATA_DIR, MODEL_DIR, LANGS

df = pd.read_csv(DATA_DIR / "combined_train.csv")
df = df.sample(n=1000, random_state=42)

dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model.to("cpu")   

def preprocess(x):
    tokenizer.src_lang = LANGS[x["src_lang"]]["nllb"]
    inp = tokenizer(x["source_text"], truncation=True)
    tgt = tokenizer(x["target_text"], truncation=True)
    inp["labels"] = tgt["input_ids"]
    return inp

dataset = dataset.map(preprocess)

args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=20,
    save_steps=100,
    report_to="none",
    no_cuda=True,   # 🔥 VERY IMPORTANT
    fp16=False,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)