import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from config import TARGET_SUBS, STOP, FIG_DIR


def run_classifier(df, save=True):
    Xtr, Xte, ytr, yte = train_test_split(df["clean"], df["subreddit"], test_size=0.25, stratify=df["subreddit"], random_state=42)

    vec = TfidfVectorizer(stop_words=STOP, sublinear_tf=True, min_df=20, ngram_range=(1, 2), max_features=30000)
    Xtr_v, Xte_v = vec.fit_transform(Xtr), vec.transform(Xte)

    clf = MultinomialNB().fit(Xtr_v, ytr)
    pred = clf.predict(Xte_v)
    print("accuracy:", round(accuracy_score(yte, pred), 3), "\n")
    print(classification_report(yte, pred))

    cm = confusion_matrix(yte, pred, labels=TARGET_SUBS, normalize="true")
    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="rocket_r", xticklabels=TARGET_SUBS, yticklabels=TARGET_SUBS, square=True, cbar_kws={"label": "share of true community", "shrink": .8})
    plt.ylabel("true community")
    plt.xlabel("predicted community")
    plt.title("Confusion matrix: where the classifier mixes communities up", pad=14)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    #plt.show()
    return clf, cm