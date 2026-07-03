import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import TARGET_SUBS, FIG_DIR


def compute_bert_profiles(df, model_name="all-MiniLM-L6-v2"):
    """Embed each post with SBERT, average into one profile vector per community."""
    model = SentenceTransformer(model_name)
    emb = model.encode(df["content"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    profiles = np.vstack([emb[df.subreddit.values == s].mean(axis=0) for s in TARGET_SUBS])
    return emb, profiles


def plot_bert_heatmap(profiles, save=True):
    sim = cosine_similarity(profiles)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="Purples",
                xticklabels=TARGET_SUBS, yticklabels=TARGET_SUBS, square=True,
                cbar_kws={"label": "cosine similarity (SBERT)", "shrink": .8})
    plt.title("Semantic similarity between communities (SBERT + cosine)", pad=14)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "bert_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    #plt.show()
    print("mean off-diagonal similarity (SBERT):")
    for i, s in enumerate(TARGET_SUBS):
        off = np.mean([sim[i, j] for j in range(len(TARGET_SUBS)) if j != i])
        print(f"  {s:22s} {off:.3f}")
    return sim