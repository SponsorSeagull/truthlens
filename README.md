# TruthLens — Fake News Detection System

> A fully client-side web application that detects fake news using **Logistic Regression** and **Support Vector Machine (SVM)** trained locally in the browser via Pyodide (Python in WebAssembly). No backend server, no external APIs, no data leaves your device.

**Live Demo → [sponsorseagull.github.io/truthlens](https://sponsorseagull.github.io/truthlens)**

---

## Screenshots

| Detect Page | Model Dashboard |
|---|---|
| Paste any news article and get an instant REAL/FAKE verdict with confidence scores | Side-by-side comparison of both models with charts and confusion matrices |

---

## Features

- **Fake News Detection** — paste any news article or headline and get a verdict with confidence percentages
- **Two ML Models** — Logistic Regression and SVM (LinearSVC with Platt scaling)
- **Model Comparison Dashboard** — Accuracy, Precision, Recall, F1 Score, ROC-AUC, Training Time, Confusion Matrices, Bar Chart, Radar Chart
- **Text Cleaning** — strips Reuters datelines, source tags, URLs, and scraping artifacts to prevent shortcut learning
- **100% Client-Side** — Python runs in the browser via Pyodide + WebAssembly. Zero backend, zero APIs
- **10,000 Records** — 5,000 fake + 5,000 real, balanced dataset from Kaggle

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn (Logistic Regression, LinearSVC) |
| Feature Extraction | TF-IDF Vectorizer (10k features, bigrams) |
| Python Runtime | Pyodide 0.25 (Python compiled to WebAssembly) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4 |
| Deployment | GitHub Pages (static, no server needed) |
| Dataset | Kaggle Fake and Real News Dataset |

---

## Project Structure

```
truthlens/
├── index.html          ← Full web app (frontend + Python ML via Pyodide)
├── model.py            ← Standalone Python script (for local execution / review)
├── README.md
└── data/
    ├── Fake.csv        ← 5,000 fake news articles (Kaggle)
    └── True.csv        ← 5,000 real news articles (Kaggle)
```

---

## How It Works

### 1. Data Loading
The app fetches `Fake.csv` and `True.csv` from the `/data/` folder directly in the browser. 5,000 records are sampled from each class for a balanced 10,000-record dataset.

### 2. Text Cleaning
Before training, each article is cleaned to remove:
- News wire datelines (`WASHINGTON (Reuters) —`)
- Inline source tags (`(AP)`, `(AFP)`, `(BBC)`)
- URLs and hyperlinks
- Scraping artifacts (`Featured Image`, `Watch:`, `Read More`)

This prevents **shortcut learning** — where the model learns to identify sources rather than writing patterns.

### 3. TF-IDF Vectorisation
Text is converted to numerical features using TF-IDF with:
- Max 10,000 features
- Unigrams + bigrams (`ngram_range=(1,2)`)
- Sublinear TF scaling (`log(tf)`)
- English stop words removed

### 4. Model Training
Both models are trained on an 80/20 stratified train-test split:

**Logistic Regression**
- Probabilistic linear classifier
- Outputs class probabilities directly
- Solver: `lbfgs`, C=1.0, max_iter=1000

**SVM (LinearSVC + Platt Scaling)**
- Maximum-margin linear classifier
- Probabilities estimated via `CalibratedClassifierCV` (3-fold CV)
- C=1.0, max_iter=2000

### 5. Inference
User input is cleaned with the same pipeline, vectorised with the trained TF-IDF, and passed to the selected model. The output is a binary prediction (REAL/FAKE) with class probability scores.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~97% | ~97% | ~97% | ~97% | ~99% |
| SVM (LinearSVC) | ~98% | ~98% | ~98% | ~98% | ~99% |

> Metrics are computed on the 20% held-out test set (2,000 records).

---

## Research Findings

### Finding 1 — Shortcut Learning
Without text cleaning, the model's strongest predictor was the word **"Reuters"** (coefficient −11.5), meaning it learned *source attribution* as a proxy for real news rather than actual writing patterns. After cleaning, this dropped to −3.8 and the top signal shifted to **"said"** — a genuine journalism pattern.

### Finding 2 — Temporal Drift
The dataset covers **2016–2018 US political news**. Articles about post-2020 events (COVID-19, 2026 elections, SpaceX IPO) are often misclassified because these topics are entirely absent from training data. This is a known limitation of static NLP classifiers.

### Finding 3 — Domain Bias
The model performs well on US political news but less accurately on other domains (UK politics, health, sports, entertainment) because the Kaggle ISOT dataset was scraped primarily from US-focused sources like Reuters and PolitiFact.

---

## Run Locally

### Option A — View the web app locally

You need a local server (Pyodide doesn't work over `file://`):

```bash
# Clone the repo
git clone https://github.com/SponsorSeagull/truthlens.git
cd truthlens

# Start a local server
python -m http.server 8000
```

Open `http://localhost:8000` in your browser.

### Option B — Run the standalone Python script

```bash
# Install dependencies
pip install pandas scikit-learn

# Run
python model.py
```

This trains both models and prints full metrics, confusion matrices, and sample predictions to the terminal.

---

## Dataset

**Kaggle Fake and Real News Dataset** by Clément Bisaillon

- Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- `Fake.csv` — ~23,000 fake articles (political news, 2016–2018)
- `True.csv` — ~21,000 real articles (Reuters, 2016–2018)
- This project uses a balanced 5,000 + 5,000 sample

---

## Guidelines Compliance

| Requirement | Status | Implementation |
|---|---|---|
| Web-based application | GitHub Pages, runs in browser |
| Minimum 2 ML algorithms | Logistic Regression + SVM |
| No external APIs | Pyodide runs Python locally in browser |
| User input with prediction | Detect page with confidence scores |
| Model comparison dashboard | Metrics table, bar chart, radar chart, confusion matrices |
| Accuracy, Precision, Recall, F1 | Shown on dashboard and detect page |
| ROC-AUC | Shown in dashboard metrics table |
| Training Time | Shown in dashboard metrics table |
| Minimum 500 records | 10,000 records used |

---

## Limitations

- **Temporal drift** — trained on 2016–2018 data; may misclassify post-2020 news
- **Domain bias** — optimised for US political news; other domains less accurate
- **First load time** — Pyodide + scikit-learn downloads ~40MB from CDN on first visit; subsequent loads use browser cache
- **No persistent model** — models retrain on every page load (browser limitation)

---

## Author

**Mustafa** — BS Computer Science  
Machine Learning Course Project
