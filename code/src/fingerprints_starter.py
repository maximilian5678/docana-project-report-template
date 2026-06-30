import os
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

TARGET_SUBREDDITS = {
    "Diablo",
    "leagueoflegends",
    "Games",
    "relationship_advice",
    "personalfinance",
    "atheism",
}

POSTS_PER_SUBREDDIT = 500
CACHE_PATH = "subset.parquet"
TOP_TERMS = 15

STOPWORDS = "english"

EXTRA_STOPWORDS = {
    "just", "like", "really", "actually", "ve", "im", "dont", "doesnt",
    "didnt", "youre", "thats", "ive", "id", "ill", "wasnt", "isnt",
    "cant", "wouldnt", "couldnt", "gonna", "wanna", "got",
}


def build_stopwords():
    """Return the stopword set for the current mode (or None to keep all words)."""
    if STOPWORDS is None:
        return None
    return list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))


def build_subset() -> pd.DataFrame:
    """Stream the dataset and keep up to POSTS_PER_SUBREDDIT posts per target sub."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached subset from {CACHE_PATH}")
        df_cached = pd.read_parquet(CACHE_PATH)
        if "text" in df_cached.columns:
            return df_cached
        print(f"Cached file {CACHE_PATH} is missing 'text' column. Rebuilding...")
        os.remove(CACHE_PATH)

    print("Streaming dataset and filtering (this runs once)...")

    stream = load_dataset(
        "parquet",
        data_files="hf://datasets/webis/tldr-17@refs/convert/parquet/default/**/*.parquet",
        split="train",
        streaming=True,
    )

    counts = {sub: 0 for sub in TARGET_SUBREDDITS}
    rows = []
    for ex in stream:
        sub = ex.get("subreddit")
        if sub in TARGET_SUBREDDITS and counts[sub] < POSTS_PER_SUBREDDIT:
            rows.append({"subreddit": sub, "text": ex.get("content", "")})
            counts[sub] += 1
        if all(c >= POSTS_PER_SUBREDDIT for c in counts.values()):
            break

    if not rows:
        raise RuntimeError(
            "No posts collected. Check that the target subreddits actually occur "
            "in the converted Parquet files (the auto-conversion only covers the "
            "first ~5 GB), or lower POSTS_PER_SUBREDDIT / pick more common subs."
        )

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_PATH)
    print("Collected posts per subreddit:")
    print(df["subreddit"].value_counts())
    return df

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cleaned_docs(df: pd.DataFrame):
    """Clean posts and concatenate all posts of a subreddit into ONE document."""
    df = df.copy()
    df["text"] = df["text"].fillna("").map(clean)
    docs = df.groupby("subreddit")["text"].apply(" ".join)
    return docs.index.tolist(), docs.values

def build_profiles(subreddits, doc_texts):
    """
    Each subreddit is ONE document (all its posts concatenated).
    The tf-idf matrix is (n_subreddits x n_terms): each row is a
    subreddit's vector in the vector space model.
    """
    vectorizer = TfidfVectorizer(
        stop_words=build_stopwords(),
        max_features=5000,
        min_df=2,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(doc_texts)
    terms = vectorizer.get_feature_names_out()
    return tfidf, terms

def print_tfidf_terms(subreddits, tfidf, terms, k=TOP_TERMS):
    print("\n=== Top terms per subreddit (tf-idf baseline) ===")
    dense = tfidf.toarray()
    for i, sub in enumerate(subreddits):
        top_idx = dense[i].argsort()[::-1][:k]
        print(f"\nr/{sub}:")
        print("  " + ", ".join(terms[j] for j in top_idx))

def print_logodds_terms(subreddits, doc_texts, k=TOP_TERMS):
    vec = CountVectorizer(stop_words=build_stopwords(), min_df=2)
    counts = vec.fit_transform(doc_texts).toarray().astype(float)
    terms = vec.get_feature_names_out()

    total = counts.sum(axis=0)
    a0 = total.sum()
    N = counts.sum() 

    print("\n=== Distinctive terms per subreddit (log-odds, informative prior) ===")
    for i, sub in enumerate(subreddits):
        y_i = counts[i]
        n_i = y_i.sum()
        y_rest = total - y_i
        n_rest = N - n_i

        log_i = np.log((y_i + total) / (n_i + a0 - y_i - total))
        log_r = np.log((y_rest + total) / (n_rest + a0 - y_rest - total))
        delta = log_i - log_r 
        var = 1.0 / (y_i + total) + 1.0 / (y_rest + total)
        z = delta / np.sqrt(var)

        top_idx = z.argsort()[::-1][:k]
        print(f"\nr/{sub}:")
        print("  " + ", ".join(terms[j] for j in top_idx))


def similarity_map(subreddits, tfidf):
    sim = cosine_similarity(tfidf)
    sim_df = pd.DataFrame(sim, index=subreddits, columns=subreddits)
    print("\n=== Cosine similarity between subreddits ===")
    print(sim_df.round(3))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(subreddits)))
    ax.set_yticks(range(len(subreddits)))
    ax.set_xticklabels(subreddits, rotation=45, ha="right")
    ax.set_yticklabels(subreddits)
    for i in range(len(subreddits)):
        for j in range(len(subreddits)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    mode = "thematic" if STOPWORDS else "stylistic"
    ax.set_title(f"Subreddit linguistic similarity ({mode})")
    fig.tight_layout()
    out = f"similarity_heatmap_{mode}.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved heatmap to {out}")
    return sim_df

def main():
    df = build_subset()
    subreddits, doc_texts = cleaned_docs(df)

    tfidf, terms = build_profiles(subreddits, doc_texts)
    print_tfidf_terms(subreddits, tfidf, terms)
    print_logodds_terms(subreddits, doc_texts)
    similarity_map(subreddits, tfidf)


if __name__ == "__main__":
    main()
