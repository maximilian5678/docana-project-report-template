import re
import pandas as pd
from config import TARGET_SUBS, N_PER_SUB, CACHE
from datasets import load_dataset
from huggingface_hub import HfFileSystem

TARGET_LC = {s.lower(): s for s in TARGET_SUBS}


def load_subset() -> pd.DataFrame:
    # Check if a cached subset exists, otherwise build it from the webis/tldr-17 dataset
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        if len(df):
            print("loaded cached subset:", df.shape)
            return df
    df = _build_subset()
    df.to_parquet(CACHE, index=False)
    print("built subset:", df.shape, "->", CACHE)
    return df


def _build_subset() -> pd.DataFrame:
    """Build a subset of the webis/tldr-17 dataset, with N_PER_SUB posts per target subreddit."""
    fs = HfFileSystem()
    shards = ["hf://" + p for p in fs.glob("datasets/webis/tldr-17@refs/convert/parquet/**/*.parquet")]
    ds = load_dataset("parquet", data_files=shards, split="train", streaming=True)

    buckets = {s: [] for s in TARGET_SUBS}
    seen = 0
    for row in ds:
        seen += 1
        canon = TARGET_LC.get(str(row.get("subreddit", "")).lower())
        if canon and len(buckets[canon]) < N_PER_SUB:
            txt = row.get("content") or row.get("body") or ""
            if len(txt.split()) >= 10:
                buckets[canon].append(txt)
        if seen % 200000 == 0:
            print(f"  scanned {seen:,} | " + ", ".join(f"{s}:{len(b)}" for s, b in buckets.items()))
        if all(len(b) >= N_PER_SUB for b in buckets.values()):
            break
    return pd.DataFrame([{"subreddit": s, "content": t} for s, ts in buckets.items() for t in ts])


def clean(s: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""
    s = s.lower().replace("'", "")
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def add_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Add a "clean" column to the dataframe, which contains the cleaned text of each post."""
    df = df.copy()
    df["clean"] = df["content"].map(clean)
    return df
