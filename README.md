# Linguistic Fingerprints of Reddit Communities

_By Carmely Reiska, Maximilian Jaeger_

## Introduction

Online communication platforms such as Reddit offer a wide field of distinct communities that are centered around specific questions, interests, beliefs, or hobbies [1]. However, what sets these subreddits apart often goes far beyond their core themes, extending into highly distinctive writing and linguistic styles. Studying these unique writing styles via stylometry allows researchers to analyze and identify the underlying structural patterns that characterize a certain text [2]. Previous research in natural language processing has shown that stylistic features are a stronger indicator of a community than the actual content that is being discussed [1]. Furthermore, as users actively engage with these forums, their writing and linguistic styles tend to align with the dominant style of their chosen community. 

This project explores this phenomenon by quantifying the linguistic differences and common language patterns among various subreddits. By applying computational text analysis and natural language processing, we aim to map the unique stylometric features of these communities. Ultimately, we want to highlight the unique text patterns of different subreddits and find out if communities with completely unrelated topics happen to share a similar way of writing.


## Dataset

The foundation of this project is the Webis-TLDR-17 corpus [3], which is a large collection of roughly 3.8 million Reddit posts written between 2006 and 2016. It was originally compiled for research on abstractive summarization, where each post consists of a main text and a short author written "TL;DR". It is labelled with its subreddit, which provides a reliable ground truth for the community a certain text belongs to. Since the posts were authored by humans and filtered for bot generated content, the material reflects genuine user writing.

For our analysis we concentrate on the content field, which captures the writing style of each author, while the summaries lie outside the scope of our research question. As the full corpus is too large to process entirely, we compiled a balanced subset of nine subreddits representing four thematic clusters, namely gaming, advice and self improvement, debate and opinion, and a general purpose community as a neutral reference point. Sampling up to 2500 posts per subreddit yields roughly 22500 evenly distributed documents. One property to keep in mind is that the corpus only contains posts with a TL;DR, which favours longer submissions and slightly biases post length.

## Methods

### Setup 

All experiments were conducted using Python 3.14.4 and open-source libraries. The analysis is entirely CPU-based and runs on a standard laptop. The main dependencies are pinned in `requirements.txt`.

To set up the environment and install all dependencies, run the following commands:

```bash
git clone https://github.com/maximilian5678/docana-project-report-template.git
cd docana-project-report-template
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/src/requirements.txt
```

It is then run with a single command:

```bash
cd code/src
python run.py
```

On the first run, the corpus is streamed and cached locally as `data/subset.parquet`. Subsequent runs will load the cached data instantly and save all generated figures to the `figures/` directory.

### Experiments

Report how you conducted the experiments. We suggest including detailed explanations of the preprocessing steps and model training in your project. For the preprocessing, describe  data cleaning, normalization, or transformation steps you applied to prepare the dataset, along with the reasons for choosing these methods. In the section on model training, explain the methodologies and algorithms you used, detail the parameter settings and training protocols, and describe any measures taken to ensure the validity of the models.

## Results and Discussion

Present the findings from your experiments, supported by visual or statistical evidence. Discuss how these results address your main research question.

## Conclusion

Summarize the major outcomes of your project, reflect on the research findings, and clearly state the conclusions you've drawn from the study.

## Contributions

| Team Member  | Contributions                                             |
|--------------|-----------------------------------------------------------|
| Carmely Reiska  | report                                                     |
| Maximilian Jaeger  | report, tf-idf similiarity                                                       |

## References

[1] [Characterizing the Language of Online Communities and its Relation to Community Reception](https://aclanthology.org/D16-1108/) (Tran & Ostendorf, EMNLP 2016)

[2] [Using Authorship Verification to Mitigate Abuse in Online Communities](https://doi.org/10.1609/icwsm.v16i1.19359) (Weerasinghe et al., ICWSM 2022)

[3] [https://huggingface.co/datasets/webis/tldr-17] (https://huggingface.co/datasets/webis/tldr-17)
