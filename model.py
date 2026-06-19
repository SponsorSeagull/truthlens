"""
TruthLens - Fake News Detection System
=======================================
Standalone Python ML script (for code review / presentation).
The same logic runs inside index.html via Pyodide (WebAssembly).

Requirements:
    pip install pandas scikit-learn

Usage:
    python model.py
"""

import re, io, json, time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             classification_report)

# ── 1. TEXT CLEANING ────────────────────────────────────────
def clean(text):
    """
    Remove news wire datelines, source tags, URLs, and
    scraping artifacts so the model learns writing style
    rather than source labels (fixes shortcut learning).
    """
    if not isinstance(text, str):
        return ''
    # Remove datelines e.g. "WASHINGTON (Reuters) -"
    text = re.sub(r'[A-Z][A-Z\s,]+\([^)]{2,20}\)\s*[-–]\s*', '', text)
    # Remove inline source tags e.g. "(Reuters)"
    text = re.sub(r'\((Reuters|AP|AFP|BBC|CNN|NPR|UPI)\)', '', text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove scraping artifacts common in fake news datasets
    text = re.sub(r'featured image|getty images?|watch\s*:|read more|click here|subscribe',
                  '', text, flags=re.IGNORECASE)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()


# ── 2. LOAD DATASET ─────────────────────────────────────────
print("Loading dataset...")
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

fake_df['label'] = 1   # 1 = Fake
true_df['label'] = 0   # 0 = Real

# Balance: 5,000 from each class (10,000 total)
df = pd.concat([
    fake_df.sample(n=min(5000, len(fake_df)), random_state=42),
    true_df.sample(n=min(5000, len(true_df)), random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Total records : {len(df)}")
print(f"  Fake (1)      : {(df['label']==1).sum()}")
print(f"  Real (0)      : {(df['label']==0).sum()}")


# ── 3. FEATURE ENGINEERING ──────────────────────────────────
# Combine title + text, apply cleaning
df['content'] = (df['title'].fillna('').apply(clean) + ' ' +
                 df['text'].fillna('').apply(clean))
df = df[df['content'].str.strip().str.len() > 20]

X = df['content'].values
y = df['label'].values


# ── 4. TRAIN / TEST SPLIT ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")


# ── 5. TF-IDF VECTORISATION ─────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True     # apply log(tf) scaling
)
X_train_v = tfidf.fit_transform(X_train)
X_test_v  = tfidf.transform(X_test)

print(f"\nTF-IDF vocabulary size : {len(tfidf.vocabulary_)}")


# ── 6. TRAIN MODELS ─────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    """Train model and print full evaluation report."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 2)

    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    acc   = round(accuracy_score(y_te, preds),  4)
    prec  = round(precision_score(y_te, preds), 4)
    rec   = round(recall_score(y_te, preds),    4)
    f1    = round(f1_score(y_te, preds),        4)
    auc   = round(roc_auc_score(y_te, proba),   4)
    cm    = confusion_matrix(y_te, preds)

    print(f"  Training Time : {elapsed}s")
    print(f"  Accuracy      : {acc*100:.1f}%")
    print(f"  Precision     : {prec*100:.1f}%")
    print(f"  Recall        : {rec*100:.1f}%")
    print(f"  F1 Score      : {f1*100:.1f}%")
    print(f"  ROC-AUC       : {auc*100:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred Real  Pred Fake")
    print(f"  Actual Real    {cm[0][0]}       {cm[0][1]}")
    print(f"  Actual Fake    {cm[1][0]}       {cm[1][1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, preds, target_names=['Real','Fake'], indent=2))

    return model

# Logistic Regression
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr = evaluate("LOGISTIC REGRESSION", lr, X_train_v, y_train, X_test_v, y_test)

# SVM (LinearSVC + Platt scaling for probabilities)
base_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm = CalibratedClassifierCV(base_svm, cv=3)
svm = evaluate("SVM (LinearSVC + Platt Scaling)", svm, X_train_v, y_train, X_test_v, y_test)


# ── 7. SAMPLE INFERENCE ─────────────────────────────────────
print("\n" + "="*50)
print("  SAMPLE PREDICTIONS")
print("="*50)

samples = [
    ("Special counsel Robert Mueller indicted 13 Russian nationals "
     "for interfering in the 2016 presidential election using social media."),
    ("BREAKING: Obama caught on secret recording admitting he deliberately "
     "weakened the US military. Pentagon officials preparing formal complaint."),
]

for text in samples:
    vec  = tfidf.transform([clean(text)])
    pred = lr.predict(vec)[0]
    prob = lr.predict_proba(vec)[0]
    label = "FAKE" if pred == 1 else "REAL"
    print(f"\n  Text   : {text[:70]}...")
    print(f"  LR     : {label}  (Real={prob[0]*100:.1f}%  Fake={prob[1]*100:.1f}%)")

    pred  = svm.predict(vec)[0]
    prob  = svm.predict_proba(vec)[0]
    label = "FAKE" if pred == 1 else "REAL"
    print(f"  SVM    : {label}  (Real={prob[0]*100:.1f}%  Fake={prob[1]*100:.1f}%)")
