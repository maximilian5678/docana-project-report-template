import math
import numpy as np
import matplotlib.pyplot as plt
from config import TARGET_SUBS, FIG_DIR

def plot_distinctive_terms(terms, profiles, k=12, cols=3, save=True):
    n_sub = len(TARGET_SUBS)
    rows = math.ceil(n_sub / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n_sub:
            ax.axis("off")
            continue
        order = np.argsort(profiles[i])[::-1][:k]
        ax.barh(terms[order][::-1], profiles[i][order][::-1], color="#3b6ea5")
        
        ax.set_title(TARGET_SUBS[i], fontweight="bold")
        ax.set_xlabel("mean tf-idf")
        
    fig.suptitle("Distinctive terms per community (tf-idf)", y=1.01, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "distinctive_terms.png", dpi=150, bbox_inches="tight")
    #plt.show()