"""Shared constants, paths and tf-idf settings."""
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# target communities (subreddits) to analyze
TARGET_SUBS = [
    "Diablo", "leagueoflegends", "Games",                 # gaming
    "relationship_advice", "personalfinance", "Fitness",  # advice / self-improvement
    "atheism", "politics",                                # debate / opinion
    "AskReddit",                                          # casual general-purpose anchor
]

# number of posts to sample per target subreddit
N_PER_SUB = 2500

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
FIG_DIR  = BASE_DIR.parent.parent / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
CACHE = DATA_DIR / "subset.parquet"

# Extended stopword list (keeps stylistically meaningful words out)
EXTRA_STOP = {"people", "think", "know", "really", "would", "like", "get", "im",
              "dont", "ive", "youre", "thats", "just", "also"}
STOP = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))

# TF-IDF settings
TFIDF_KW = dict(stop_words=STOP, sublinear_tf=True, min_df=20, max_features=20000)
