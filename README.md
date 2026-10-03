# Sentiment Analysis on Three Review Domains

A reproducible binary sentiment classifier built on the [UCI Sentiment Labelled Sentences dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) (CC BY 4.0). It contains 3,000 labeled sentences, 1,000 each from Amazon, IMDb, and Yelp. Label `1` means positive; `0` means negative. The script downloads the original archive on first run; dataset text is not included in this repository.

## What the project does

The script combines all three sources, removes empty rows, conflicting labels for identical texts, and duplicates, then creates a stratified 64% train / 16% validation / 20% test split (final counts can vary after cleaning). It compares TF-IDF + logistic regression with TF-IDF + a one-hidden-layer multilayer perceptron (MLP). Both are trained only on the train split. The architecture with the higher validation F1 is refit on train + validation and evaluated once on the held-out test split. This is a neural-network experiment; the selected final model may be the baseline if it performs better. The tokenizer/vectorizer is fit within each pipeline, preventing test vocabulary leakage.

## Run locally (Python 3.10+)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python predict.py "The service was excellent and delivery was fast."
```

On Google Colab, upload `train.py`, `predict.py`, and `requirements.txt`, then run `!pip install -r requirements.txt` and `!python train.py`. Download the `outputs` directory afterward. Internet access is needed for the first dataset download. Alternatively place the original UCI zip at `data/sentiment_labelled_sentences.zip`.

## Output and evaluation

`outputs/metrics.json`: split counts, validation scores for both models, selected model, test accuracy, precision, recall, F1 and ROC-AUC, and per-class report. `outputs/confusion_matrix.png`: test confusion matrix. `outputs/test_predictions.csv`: review text, source, true/predicted label and probability. `outputs/sentiment_model.joblib`: trained inference pipeline.

Accuracy alone can hide uneven errors; F1 balances precision and recall. Selection uses validation F1; test performance is reported only after selection. With a small dataset, estimates may vary across splits and performance on unrelated domains is uncertain. A TF-IDF MLP does not model word order as an LSTM or transformer would.

## Repository submission checklist

- Upload source files, README, `requirements.txt`, and the `outputs` folder after running training.
- Add your screenshots of terminal training output, `metrics.json`, confusion matrix, and a successful `predict.py` command to a `screenshots` folder.
- Export a short PDF covering objective, dataset citation and license, preprocessing/splits, model comparison, actual metrics, confusion matrix, limitations, and your personal learning.
- Do not claim measured accuracy until the run has finished. Avoid uploading `.venv` or the dataset archive.

## Attribution

Dataset: Kotzias et al., *From Group to Individual Labels Using Deep Features*, KDD 2015, via UCI Machine Learning Repository. Licensed CC BY 4.0. Code in this repository is original project implementation.
