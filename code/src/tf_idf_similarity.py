import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import TARGET_SUBS, TFIDF_KW, FIG_DIR


def compute_profiles(df):
    """Fit tf-idf on all posts, average into one profile vector per community."""
    vec = TfidfVectorizer(**TFIDF_KW)
    X = vec.fit_transform(df["clean"])
    terms = np.array(vec.get_feature_names_out())
    profiles = np.vstack([np.asarray(X[df.subreddit.values == s].mean(axis=0)).ravel() for s in TARGET_SUBS])
    return vec, X, terms, profiles


def plot_heatmap(profiles, save=True):
    """Plot cosine similarity heatmap of the per-community tf-idf profiles."""
    sim = cosine_similarity(profiles)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="Purples", xticklabels=TARGET_SUBS, yticklabels=TARGET_SUBS, square=True, cbar_kws={"label": "cosine similarity", "shrink": .8})
    plt.title("Linguistic similarity between communities (tf-idf + cosine)", pad=14)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "similarity_heatmap.png", dpi=150, bbox_inches="tight")
    #plt.show()
    print("mean off-diagonal similarity (lower = outlier):")
    for i, s in enumerate(TARGET_SUBS):
        off = np.mean([sim[i, j] for j in range(len(TARGET_SUBS)) if j != i])
        print(f"  {s:22s} {off:.3f}")
    return sim
