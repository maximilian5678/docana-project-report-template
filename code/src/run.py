"""
End-to-end pipeline for analyzing subreddit content.
Plots can be found in the figures/ folder.
"""
from data import load_subset, add_clean
from similarity import compute_profiles, plot_heatmap
#from distinctive_terms import plot_distinctive_terms
#from sentiment import add_sentiment, plot_sentiment
#from classification import run_classifier


def main():
    df = load_subset()
    df = add_clean(df)

    vec, X, terms, profiles = compute_profiles(df)
    plot_heatmap(profiles)
    #plot_distinctive_terms(terms, profiles)

    #df = add_sentiment(df)
    #plot_sentiment(df)
    #run_classifier(df)


if __name__ == "__main__":
    main()
