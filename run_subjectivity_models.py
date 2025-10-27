import os
import sys
import time
import pandas as pd
from tqdm import tqdm
from transformers import (
    pipeline,
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datetime import datetime

MODELS = {
    "GroNLP_mdebertav3_subjectivity_english": "GroNLP/mdebertav3-subjectivity-english",
    "MatteoFasulo_mdeberta_v3_base_subjectivity_sentiment_english": "MatteoFasulo/mdeberta-v3-base-subjectivity-sentiment-english",
    "cffl_bert_base_styleclassification_subjective_neutral": "cffl/bert-base-styleclassification-subjective-neutral",
    "Gladiator_microsoft_deberta_v3_large_cls_subj": "Gladiator/microsoft-deberta-v3-large_cls_subj",
}
# Models assumed English-only
ENGLISH_ONLY = {
    "cffl_bert_base_styleclassification_subjective_neutral",
    "Gladiator_microsoft_deberta_v3_large_cls_subj",
}

TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-ru-en"

# --- Runtime / HF hub configuration & simple logger ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid deadlocks / noisy warnings
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster Rust downloader when available

def LOG(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_translator(retries: int = 3, sleep_base: float = 1.5):
    LOG("Loading MarianMT translator tokenizer & model...")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            tokenizer = MarianTokenizer.from_pretrained(TRANSLATOR_MODEL)
            model = MarianMTModel.from_pretrained(TRANSLATOR_MODEL)
            LOG("Translator loaded.")
            return tokenizer, model
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            LOG(f"Translator load failed (attempt {attempt}/{retries}): {e}")
            time.sleep(sleep_base ** attempt)
    raise RuntimeError(f"Failed to load translator after {retries} attempts: {last_err}")


def translate_texts(texts, tokenizer, model, batch_size=16):
    LOG(f"Translating {len(texts)} texts ru→en in batches of {batch_size}...")
    translated = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translate", unit="batch"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translated.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    LOG("Translation done.")
    return translated


def _load_classifier(model_name: str, tokenizer_name: str | None = None, retries: int = 3, sleep_base: float = 1.5):
    if tokenizer_name is None:
        tokenizer_name = model_name
    LOG(f"Loading classifier: model={model_name} tokenizer={tokenizer_name}")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True, use_fast=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, ignore_mismatched_sizes=True, trust_remote_code=True
            )
            LOG("Classifier loaded.")
            return tokenizer, model
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            LOG(f"Classifier load failed (attempt {attempt}/{retries}): {e}")
            time.sleep(sleep_base ** attempt)
    raise RuntimeError(f"Failed to load classifier after {retries} attempts: {last_err}")


def classify_texts(texts, model_name, batch_size: int = 8):
    tokenizer_name = (
        "microsoft/mdeberta-v3-base"
        if model_name == "MatteoFasulo/mdeberta-v3-base-subjectivity-sentiment-english"
        else model_name
    )
    tokenizer, model = _load_classifier(model_name, tokenizer_name)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

    LOG(f"Classifying {len(texts)} texts with {model_name} (batch_size={batch_size})...")
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Infer", unit="batch"):
        batch = texts[i:i + batch_size]
        out = clf(batch, batch_size=batch_size, truncation=True)
        preds.extend(out)
    LOG("Classification done.")
    return [p["label"] for p in preds]


def process_dataset(path, text_col, date_col, prefix, limit=None):
    df = pd.read_excel(path)
    if limit is not None:
        df = df.head(limit)
    df[text_col] = df[text_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    translator_tok = translator_mod = None

    all_aggs = []
    for alias, model_id in tqdm(list(MODELS.items()), desc=f"Models[{prefix}]", unit="model"):
        LOG(f"\n--- [{prefix}] Starting model '{alias}' → {model_id}")
        texts = df[text_col].tolist()
        if prefix == "RBK" and alias in ENGLISH_ONLY:
            LOG("Detected English-only model; translating Russian texts...")
            if translator_tok is None:
                translator_tok, translator_mod = load_translator()
            texts = translate_texts(texts, translator_tok, translator_mod)
        labels = classify_texts(texts, model_id)
        df[f"{alias}_label"] = labels
        LOG("Aggregating daily counts by label...")
        for label in set(labels):
            col = f"{alias}_{prefix}_{label}"
            df[col] = (df[f"{alias}_label"] == label).astype(int)
            agg = df.groupby(date_col)[col].sum()
            all_aggs.append(agg)
        LOG(f"Completed model '{alias}'.")
    aggregated = pd.concat(all_aggs, axis=1).reset_index().rename(columns={date_col: "date"})
    return df, aggregated


def main(limit=None):
    LOG(f"Starting run. limit={limit}")

    LOG("Processing RBK dataset...")
    rbk_df, rbk_agg = process_dataset(
        "/Users/vastepanov/PycharmProjects/Fucking_shit_subjectivity/input_files/RBK_v1.xlsx", "title", "published", "RBK", limit=limit
    )
    LOG("Processing Bloomberg dataset...")
    bmbg_df, bmbg_agg = process_dataset(
        "/Users/vastepanov/PycharmProjects/Fucking_shit_subjectivity/input_files/bloomberg_df_ver1.1.xlsx", "Title", "published date", "BMBG", limit=limit
    )

    LOG("Saving classified per-article outputs...")
    rbk_df.to_excel("/Users/vastepanov/PycharmProjects/Fucking_shit_subjectivity/input_files/RBK_v1_classified.xlsx", index=False)
    bmbg_df.to_excel("/Users/vastepanov/PycharmProjects/Fucking_shit_subjectivity/input_files/bloomberg_df_ver1.1_classified.xlsx", index=False)

    LOG("Merging with super dataset and writing final excel...")
    super_df = pd.read_excel("/Users/vastepanov/PycharmProjects/Fucking_shit_subjectivity/input_files/dataset_superfull_604.xlsx")
    super_df["date"] = pd.to_datetime(super_df["date"]).dt.date
    merged = super_df.merge(rbk_agg, on="date", how="left").merge(bmbg_agg, on="date", how="left")
    merged = merged.fillna(0)
    merged.to_excel("/Users/vastepanov/PycharmProjects/Fucking_shit_subjectivity/input_files/dataset_superfull_409.xlsx", index=False)
    LOG("All done.")


if __name__ == "__main__":
    max_rows = os.environ.get("MAX_ROWS")
    limit = int(max_rows) if max_rows else None
    main(limit=limit)
