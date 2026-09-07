
import os
import re
import pandas as pd
from datasets import load_dataset

LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}


def remove_urls(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text.strip()

def remove_twitter_artifacts(text):
    text = re.sub(r"^RT @\w+:\s*", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("&amp;", "and")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = re.sub(r"#(\w+)", r"\1", text)
    return text.strip()

def fix_encoding(text):
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def normalize_tickers(text):
    def uppercase_ticker(match):
        return match.group(0).upper()
    text = re.sub(r"\$[a-zA-Z]{1,6}\b", uppercase_ticker, text)
    return text

def lowercase_except_tickers(text):
    tickers = re.findall(r"\$[A-Z]{1,6}\b", text)
    placeholders = {}
    for i, ticker in enumerate(tickers):
        placeholder = f"__ticker{i}__"
        placeholders[placeholder] = ticker
        text = text.replace(ticker, placeholder, 1)
    text = text.lower()
    for placeholder, ticker in placeholders.items():
        text = text.replace(placeholder, ticker)
    return text

def remove_special_chars(text):
    text = re.sub(r"[^a-zA-Z0-9\s\$\%\.\,\-\']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def full_clean_pipeline(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\n\t\r]", " ", text)
    text = remove_urls(text)
    text = remove_twitter_artifacts(text)
    text = fix_encoding(text)
    text = normalize_tickers(text)
    text = lowercase_except_tickers(text)
    text = remove_special_chars(text)
    return text.strip()


def load_and_clean(output_file="financial_news.csv"):
    print("Step 1: Loading dataset from HuggingFace...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    df = dataset["train"].to_pandas()
    print(f"  Loaded {len(df)} raw rows")
    print(f"  Columns: {df.columns.tolist()}")

    print("\nStep 2: Applying cleaning pipeline...")
    df["cleaned_text"] = df["text"].astype(str).apply(full_clean_pipeline)
    df["label"] = df["label"].astype(int)
    df["sentiment_name"] = df["label"].map(LABEL_MAP)

    print("\nStep 3: Removing duplicates and empty rows...")
    before = len(df)
    df = df.drop_duplicates(subset=["cleaned_text"])
    df = df[df["cleaned_text"].str.strip() != ""]
    df = df[df["cleaned_text"].str.split().str.len() >= 3]
    df = df.reset_index(drop=True)
    after = len(df)
    print(f"  Rows before: {before}")
    print(f"  Rows after:  {after}")
    print(f"  Removed:     {before - after}")

    print("\nStep 4: Label distribution after cleaning...")
    for lbl, count in df["label"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"  {LABEL_MAP[lbl]}: {count} ({pct:.1f}%)")

    print(f"\nStep 5: Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} rows to {output_file}")
    print(f"  File size: {os.path.getsize(output_file):,} bytes")

    return df


if __name__ == "__main__":
    print("Financial News Sentiment — Data Ingestion Pipeline")
    print("=" * 55)
    df = load_and_clean()
    print("\nIngestion complete.")
    print(f"Output file: financial_news.csv")
    print(f"Total rows:  {len(df)}")
