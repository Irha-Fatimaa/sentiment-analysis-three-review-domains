"""Train reproducible baseline and neural sentiment classifiers on UCI reviews."""
import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

URL = 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'
SEED = 42


def load_data(cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'sentiment_labelled_sentences.zip'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    frames = []
    with zipfile.ZipFile(archive) as z:
        for source, filename in [('amazon', 'amazon_cells_labelled.txt'),
                                 ('imdb', 'imdb_labelled.txt'), ('yelp', 'yelp_labelled.txt')]:
            member = next(n for n in z.namelist() if n.endswith('/' + filename))
            with z.open(member) as handle:
                frame = pd.read_csv(handle, sep='\t', header=None, names=['text', 'label'],
                                    quoting=3, encoding='utf-8')
            frame['source'] = source
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=['text', 'label'])
    data['text'] = data['text'].astype(str).str.strip()
    data = data[data['text'].ne('')]
    data['label'] = data['label'].astype(int)
    if not data['label'].isin([0, 1]).all():
        raise ValueError('Unexpected labels')
    # Remove identical texts before splitting so duplicate reviews cannot cross splits.
    conflicts = data.groupby('text')['label'].nunique()
    data = data[~data['text'].isin(conflicts[conflicts > 1].index)]
    return data.drop_duplicates(subset=['text']).reset_index(drop=True)


def scores(model, texts, labels):
    pred = model.predict(texts)
    prob = model.predict_proba(texts)[:, 1]
    return dict(accuracy=accuracy_score(labels, pred),
                precision=precision_score(labels, pred, zero_division=0),
                recall=recall_score(labels, pred, zero_division=0),
                f1=f1_score(labels, pred, zero_division=0),
                roc_auc=roc_auc_score(labels, prob))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--cache', type=Path, default=Path('data'))
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    df = load_data(args.cache)
    # Stratify by label AND review source to preserve domain and class proportions.
    strata = df['source'] + '_' + df['label'].astype(str)
    trainval, test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=strata)
    strata_tv = trainval['source'] + '_' + trainval['label'].astype(str)
    train, val = train_test_split(trainval, test_size=0.20, random_state=SEED, stratify=strata_tv)
    vectorizer = dict(ngram_range=(1, 2), min_df=2, max_features=10000,
                      sublinear_tf=True, strip_accents='unicode')
    candidates = {
        'logistic_regression': Pipeline([('tfidf', TfidfVectorizer(**vectorizer)),
                                         ('classifier', LogisticRegression(max_iter=1000, random_state=SEED))]),
        'neural_mlp': Pipeline([('tfidf', TfidfVectorizer(**vectorizer)),
                                ('classifier', MLPClassifier(hidden_layer_sizes=(64,),
                                                             alpha=0.01, early_stopping=True,
                                                             n_iter_no_change=10,
                                                             max_iter=100, random_state=SEED))])
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(train['text'], train['label'])
        validation[name] = scores(model, val['text'], val['label'])
    winner = max(candidates, key=lambda n: validation[n]['f1'])
    # Refit selected architecture on train + validation; test remains untouched.
    best = candidates[winner]
    best.fit(trainval['text'], trainval['label'])
    test_metrics = scores(best, test['text'], test['label'])
    predictions = best.predict(test['text'])
    report = dict(dataset='UCI Sentiment Labelled Sentences', seed=SEED,
                  original_rows=3000, cleaned_rows=len(df), splits={
                      'train': len(train), 'validation': len(val), 'test': len(test)},
                  validation=validation, selection_metric='validation F1',
                  selected_model=winner, test=test_metrics,
                  classification_report=classification_report(test['label'], predictions,
                                                              target_names=['negative', 'positive'],
                                                              output_dict=True))
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    joblib.dump(best, args.output / 'sentiment_model.joblib')
    pd.DataFrame({'text': test['text'], 'source': test['source'],
                  'actual': test['label'], 'predicted': predictions,
                  'positive_probability': best.predict_proba(test['text'])[:, 1]}).to_csv(
                      args.output / 'test_predictions.csv', index=False)
    ConfusionMatrixDisplay.from_predictions(test['label'], predictions,
                                            display_labels=['negative', 'positive'], cmap='Blues')
    plt.title('Selected model: held-out test set')
    plt.tight_layout()
    plt.savefig(args.output / 'confusion_matrix.png', dpi=160)
    plt.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
