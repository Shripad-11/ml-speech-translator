import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR, DATA_DIR

def process(folder, lang, limit=8000):
    en_file = folder / "train.en"
    tgt_file = list(folder.glob("train.*"))[1]

    data = []

    with open(en_file, encoding="utf-8") as f_en, open(tgt_file, encoding="utf-8") as f_tgt:
        for i, (en, tgt) in enumerate(zip(f_en, f_tgt)):
            if i >= limit:
                break

            en, tgt = en.strip(), tgt.strip()
            if not en or not tgt:
                continue

            data.append([tgt, en, lang, "en"])
            data.append([en, tgt, "en", lang])

    return data

def main():
    all_data = []

    for folder in RAW_DATA_DIR.iterdir():
        lang = folder.name.split("-")[1]
        if lang in ["hi", "mr", "pa"]:
            all_data.extend(process(folder, lang))

    df = pd.DataFrame(all_data, columns=["source_text","target_text","src_lang","tgt_lang"])
    df.to_csv(DATA_DIR / "combined_train.csv", index=False)

if __name__ == "__main__":
    main()