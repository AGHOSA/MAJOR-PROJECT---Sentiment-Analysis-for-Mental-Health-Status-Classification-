# 🧠 MindLens XAI — Explainable AI for Mental Health Text Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg?logo=sqlite)](https://www.sqlite.org/)
[![XAI](https://img.shields.io/badge/Explainability-Token%20Attribution-6366F1.svg)](#-explainable-ai-xai-architecture)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MindLens XAI** is an Explainable Artificial Intelligence system and interactive web platform designed to analyze textual statements, detect indicators of **7 clinically recognized mental health conditions**, provide transparent **word-level token attribution weights**, and persist complete assessment histories with user feedback in an embedded **SQLite Database**.

---

## 📑 Table of Contents
- [📌 Project Overview & Research Context](#-project-overview--research-context)
- [🎯 The 7 Mental Health Categories](#-the-7-mental-health-categories)
- [🔬 Model Benchmarks & Comparison](#-model-benchmarks--comparison)
- [🔍 Explainable AI (XAI) Architecture](#-explainable-ai-xai-architecture)
- [🗄️ Database Architecture & Schema](#️-database-architecture--schema)
- [📡 REST API Documentation](#-rest-api-documentation)
- [💻 Web Interface & UI Features](#-web-interface--ui-features)
- [📂 Project Directory Structure](#-project-directory-structure)
- [🚀 Quick Start & Installation](#-quick-start--installation)
- [🧪 Running Automated Tests](#-running-automated-tests)
- [⚠️ Clinical & Ethical Disclaimer](#️-clinical--ethical-disclaimer)
- [📚 Research Foundation & Citation](#-research-foundation--citation)

---

## 📌 Project Overview & Research Context

Mental health conditions—such as depression, severe stress, anxiety, and suicidal ideation—affect hundreds of millions globally. While individuals increasingly express psychological states through online text, traditional "black-box" AI classifiers output categorical predictions without explanation, creating significant trust barriers for clinical screening and early intervention.

**MindLens XAI** bridges this gap by combining:
1. **High-Performance NLP & Machine Learning**: Multi-class classification using optimized TF-IDF (1, 2 n-grams, 5,000 features) paired with regularized Logistic Regression, Linear SVM, and Random Forest.
2. **Transparent Word-Level Feature Attribution**: Mathematical log-odds decomposition revealing exact positive and negative token contribution weights for every individual prediction.
3. **Embedded Persistent SQLite Database (`data/mental_health_xai.db`)**: Auto-saving all inferences, confidence scores, multi-class probability vectors, and token weights with audit logs and human-in-the-loop feedback mechanisms.
4. **Interactive Dark Glassmorphism UI**: Real-time token highlighting, preset clinical scenarios, dynamic probability distributions, and instant history inspection.

---

## 🎯 The 7 Mental Health Categories

The model is trained on a comprehensive dataset of **53,043 text samples** spanning 7 clinically distinct categories:

| Category | Dataset Share | Primary Linguistic Markers & Indicators |
| :--- | :---: | :--- |
| **Normal** | `30.83%` | Balanced emotional expression, daily productivity, optimism, absence of acute distress. |
| **Depression** | `29.04%` | Melancholy, emotional numbness, loss of motivation, feelings of emptiness, lethargy. |
| **Suicidal** | `20.08%` | Urgent crisis expressions, severe hopelessness, perceived burdensomeness, unbearable agony. |
| **Anxiety** | `7.33%` | Hyper-vigilance, panic sensations, racing heart, constant dread, overwhelming nervousness. |
| **Bipolar** | `5.42%` | Cyclical extremes between manic euphoria/grandiosity and debilitating depressive crashes. |
| **Stress** | `5.03%` | Work burnout, impossible deadlines, chronic pressure, cognitive fatigue, financial anxiety. |
| **Personality Disorder** | `2.26%` | Identity instability, severe fear of abandonment, extreme emotional dysregulation. |

---

## 🔬 Model Benchmarks & Comparison

Based on extensive empirical evaluation and hyperparameter grid-search optimization:

| Model Architecture | Baseline Acc. | Optimized Acc. | Key Hyperparameters | Training Time | Feature Interpretability |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Logistic Regression (Active)** | 75.00% | **76.44%** | `C=4.28`, `penalty='l2'`, `solver='lbfgs'` | **~3.4 min** | **High** (Direct linear log-odds) |
| **Linear SVM** | 75.00% | **76.93%** | `C=0.234`, `loss='squared_hinge'` | ~16.5 min | **High** (Hyperplane weights) |
| **Random Forest** | 74.00% | **75.37%** | `n_estimators=500`, `max_depth=None` | ~135.9 min | **Medium** (Gini feature importances) |
| **CNN + BiLSTM** | — | **77.00%** | Spatial convolutions + Bi-directional LSTM | ~15.0 min | **Low** (Attention maps) |
| **DistilBERT** | — | **80.48%** | Transformer self-attention (6-layer distilled) | ~45.0 min | **Medium** (Integrated Gradients / SHAP) |

> **Active Baseline Pipeline**: Tuned **Logistic Regression with TF-IDF** is deployed as the primary engine for its optimal balance of high accuracy (76.44%), sub-millisecond inference latency, and exact token explainability.

---

## 🔍 Explainable AI (XAI) Architecture

Traditional NLP models provide only a label (e.g. *"Depression: 92%"*). MindLens decomposes the decision boundary into exact token-level influence:

$$\text{Logit}_c(\mathbf{x}) = \beta_{0, c} + \sum_{i=1}^{M} \beta_{i, c} \cdot \text{TF-IDF}(w_i)$$

Where:
- $\beta_{i, c} > 0$: The word $w_i$ **positively contributes** toward predicting category $c$ (highlighted in emerald green).
- $\beta_{i, c} < 0$: The word $w_i$ **opposes** category $c$ (highlighted in rose red).

```
"I feel completely exhausted, hopeless, and cannot focus on anything."
         ▲            ▲          ▲
      [+0.84]      [+1.42]    [+0.65]  --> Strongly drives "Depression"
```

---

## 🗄️ Database Architecture & Schema

MindLens includes a zero-configuration, thread-safe embedded **SQLite Database** (`data/mental_health_xai.db`) operating in **WAL (Write-Ahead Logging)** mode.

### Table: `predictions`
```sql
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_text TEXT NOT NULL,
    predicted_category TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities_json TEXT NOT NULL,
    top_contributing_words_json TEXT NOT NULL,
    model_name TEXT NOT NULL,
    top_n_requested INTEGER DEFAULT 6,
    user_feedback TEXT DEFAULT NULL,
    feedback_notes TEXT DEFAULT NULL,
    corrected_category TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictions(predicted_category);
```

### Database Capabilities:
- **Automatic In-Line Persistence**: Each `/predict` API call saves the assessment record and returns the assigned `id`.
- **Audit & History Retrieval**: Paginated queries with category filtering and full-text keyword search.
- **Human-in-the-Loop Feedback Loop**: Record user validation (`accurate` / `inaccurate`) and clinician notes.
- **Real-Time Analytics Aggregation**: Direct computation of category distributions, average confidence rates, and user agreement metrics.

---

## 📡 REST API Documentation

### 1. Predict Mental Health & Explainability
**`POST /predict`**

*Request Body:*
```json
{
  "text": "My heart is pounding rapidly, I can barely breathe, and I have a constant impending dread about everything.",
  "top_n_words": 6
}
```

*Response (`200 OK`):*
```json
{
  "id": 1,
  "text": "My heart is pounding rapidly, I can barely breathe, and I have a constant impending dread about everything.",
  "predicted_category": "Anxiety",
  "confidence": 0.9412,
  "probabilities": {
    "Anxiety": 0.9412,
    "Stress": 0.0315,
    "Depression": 0.0152,
    "Normal": 0.0071,
    "Bipolar": 0.0032,
    "Personality disorder": 0.0011,
    "Suicidal": 0.0007
  },
  "top_contributing_words": [
    {"word": "dread", "score": 1.4821, "is_positive": true},
    {"word": "pounding", "score": 1.1205, "is_positive": true},
    {"word": "breathe", "score": 0.9840, "is_positive": true},
    {"word": "rapidly", "score": 0.7632, "is_positive": true},
    {"word": "constant", "score": 0.6120, "is_positive": true},
    {"word": "impending", "score": 0.5891, "is_positive": true}
  ],
  "model_name": "logistic_regression",
  "timestamp": "2026-08-30T13:48:03.123456+00:00"
}
```

---

### 2. Fetch Stored History
**`GET /api/history?limit=20&offset=0&category=Depression`**

*Response (`200 OK`):*
```json
{
  "total": 42,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "id": 14,
      "statement_text": "Feeling completely drained and cannot get out of bed.",
      "predicted_category": "Depression",
      "confidence": 0.965,
      "probabilities": { "Depression": 0.965, "Stress": 0.021 },
      "top_contributing_words": [{ "word": "drained", "score": 1.25, "is_positive": true }],
      "model_name": "logistic_regression",
      "user_feedback": "accurate",
      "created_at": "2026-08-30T13:40:12"
    }
  ]
}
```

---

### 3. Record User Feedback
**`POST /api/history/{id}/feedback`**

*Request Body:*
```json
{
  "user_feedback": "accurate",
  "feedback_notes": "Clinical statement matches major depressive disorder symptom profile.",
  "corrected_category": null
}
```

---

### 4. Fetch Aggregate Analytics
**`GET /api/analytics`**

*Response (`200 OK`):*
```json
{
  "total_predictions": 128,
  "average_confidence": 0.9124,
  "category_distribution": {
    "Depression": { "count": 48, "avg_confidence": 0.9312 },
    "Anxiety": { "count": 32, "avg_confidence": 0.9045 },
    "Normal": { "count": 24, "avg_confidence": 0.9521 },
    "Suicidal": { "count": 12, "avg_confidence": 0.8950 },
    "Stress": { "count": 8, "avg_confidence": 0.8640 },
    "Bipolar": { "count": 3, "avg_confidence": 0.8410 },
    "Personality disorder": { "count": 1, "avg_confidence": 0.7900 }
  },
  "feedback_distribution": {
    "accurate": 45,
    "inaccurate": 2
  }
}
```

---

### Complete Endpoint Summary
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves interactive web UI |
| `POST` | `/predict` | Predict mental health condition, return XAI token weights, and auto-persist to DB |
| `GET` | `/api/history` | Paginated assessment history with category filter and search |
| `GET` | `/api/history/{id}` | Full details of a single assessment |
| `DELETE` | `/api/history/{id}` | Delete a single assessment from DB |
| `DELETE` | `/api/history` | Clear all historical assessment records |
| `POST` | `/api/history/{id}/feedback` | Submit validation feedback (accurate / inaccurate) |
| `GET` | `/api/analytics` | Statistical distribution & confidence metrics |
| `GET` | `/api/health` | Health status and DB connectivity check |
| `GET` | `/api/info` | Model metadata, feature matrix params, and top global class terms |
| `GET` | `/api/examples` | Clinical preset scenarios for UI testing |

---

## 💻 Web Interface & UI Features

- **Empathetic Glassmorphism Theme**: Curated dark palette (`#0a0e17` background with radial ambient glows, translucent backdrop blurs, and vibrant condition badges).
- **Interactive Token Cloud**: Live highlighting of influential words with hover tooltips displaying exact log-odds contribution values.
- **Real-Time Probability Meters**: Animated distribution bars illustrating multi-class certainty across all 7 categories.
- **Live Database History Drawer**: View, inspect, replay, delete, and evaluate past assessments without page refresh.
- **Model Architecture Inspector**: Global top vocabulary keywords for each condition.
- **Accessibility & Productivity**: Preset one-click clinical scenarios, character countdown, and keyboard shortcut (`Ctrl` + `Enter`).

---

## 📂 Project Directory Structure

```
explainable-ai-mental-health/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI server, REST routes & endpoints
├── data/
│   ├── ml.dataset.csv              # Machine learning training dataset (53,043 samples)
│   ├── dl.dataset.csv              # Deep learning training dataset
│   └── mental_health_xai.db        # SQLite database (auto-created on startup)
├── saved_models/
│   ├── best_model/                 # Serialized tuned ML model & TF-IDF vectorizer
│   │   ├── config.json
│   │   └── model.joblib
│   └── model_comparison_summary.json
├── src/
│   ├── __init__.py
│   ├── db/
│   │   ├── __init__.py             # Database interface exports
│   │   └── database.py             # SQLite thread-safe manager & CRUD services
│   ├── models/
│   │   ├── __init__.py
│   │   └── ml_models.py            # ML classifier pipeline & explainability engine
│   └── utils/
│       ├── __init__.py
│       └── text_preprocessing.py   # Regex sanitization & text cleaning utilities
├── static/
│   ├── css/
│   │   └── style.css               # Modern glassmorphic dark theme stylesheet
│   └── js/
│       └── app.js                  # Frontend interactive application logic
├── templates/
│   └── index.html                  # Jinja2 HTML5 web application template
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 # API route & endpoint unit tests
│   ├── test_database.py            # SQLite CRUD & analytics unit tests
│   └── test_ml_models.py           # ML pipeline & text preprocessing tests
├── README.md                       # Comprehensive project documentation
└── pytest.ini                      # Test runner configuration
```

---

## 🚀 Quick Start & Installation

### 1. Clone Repository
```bash
git clone https://github.com/Rekha-1kumari/explainable-ai-mental-health.git
cd explainable-ai-mental-health
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas numpy jinja2 pytest
```

### 4. Launch the Server
```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser for the web interface.
- Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** for the interactive Swagger API documentation.

---

## 🧪 Running Automated Tests

Run the full pytest test suite across models, database, and API endpoints:

```bash
python -m pytest -v
```

Expected output:
```
tests/test_api.py::test_serve_ui PASSED                                  [ 20%]
tests/test_api.py::test_health_check PASSED                              [ 40%]
tests/test_api.py::test_predict_and_db_persistence PASSED                [ 60%]
tests/test_database.py::test_init_and_save_prediction PASSED             [ 70%]
tests/test_database.py::test_get_history_and_filtering PASSED            [ 80%]
tests/test_database.py::test_feedback_and_analytics PASSED               [ 90%]
tests/test_ml_models.py::test_ml_classifier_fit_predict_save_load PASSED [100%]

======================= 14 passed in 5.99s =======================
```

---

## ⚠️ Clinical & Ethical Disclaimer

> [!IMPORTANT]
> **This software is an AI research prototype intended strictly for linguistic analysis and research demonstrations.**
> - It is **not a diagnostic medical device** and should not be used as a substitute for professional clinical psychiatric assessment, diagnosis, or treatment.
> - If you or someone you know is in distress or experiencing thoughts of self-harm, please reach out immediately:
>   - **United States**: Call or text **988** (Suicide & Crisis Lifeline) or text `HOME` to **741741** (Crisis Text Line).
>   - **India**: Call **9152987821** (KIRAN / Vandrevala Foundation) or **14416** (Tele-MANAS).
>   - **International**: Find resources at [Befrienders Worldwide](https://www.befrienders.org/) or [Find A Helpline](https://findahelpline.com/).

---

## 📚 Research Foundation & Citation

This implementation builds upon comparative research in computational psychiatry and natural language processing:

> Sikder, M. K. (2025). *Comparative Analysis of Machine Learning and Deep Learning Models for Depression Detection Using NLP*. Journal of Emerging Technologies and Innovative Research (JETIR), 12(5), j589–j600. [JETIR2505A54](https://www.jetir.org/).

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
