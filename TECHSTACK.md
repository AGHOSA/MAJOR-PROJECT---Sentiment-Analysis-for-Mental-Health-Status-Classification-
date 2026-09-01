# 🛠️ MindLens XAI — Complete Technology Stack

A comprehensive reference of all technologies, frameworks, libraries, mathematical models, databases, and development tools utilized in the **MindLens XAI (Explainable AI for Mental Health Status Classification)** system.

---

## 📑 Table of Contents
1. [Overview & Architectural Summary](#1-overview--architectural-summary)
2. [Frontend Architecture (UI / UX)](#2-frontend-architecture-ui--ux)
3. [Backend & REST API Layer](#3-backend--rest-api-layer)
4. [Machine Learning & NLP Pipeline](#4-machine-learning--nlp-pipeline)
5. [Explainable AI (XAI) Engine](#5-explainable-ai-xai-engine)
6. [Database & Persistence Layer](#6-database--persistence-layer)
7. [Testing & Quality Assurance](#7-testing--quality-assurance)
8. [Data Processing & Benchmarking Tools](#8-data-processing--benchmarking-tools)
9. [Development & DevOps Tools](#9-development--devops-tools)
10. [Complete Stack Summary Table](#10-complete-stack-summary-table)
11. [Presentation & Voice-Over Talking Points](#11-presentation--voice-over-talking-points)

---

## 1. Overview & Architectural Summary

MindLens XAI is a multi-tier Explainable AI web application designed for multi-class mental health text screening. It operates completely self-hosted, locally, and without third-party proprietary API dependencies.

```
[ User Browser ]
       │
       ▼  (HTTP / REST JSON)
[ FastAPI + Uvicorn ASGI Server ]
       ├──► [ NLP & Preprocessing ]
       ├──► [ TF-IDF Vectorizer (1-2 N-Grams, 5k Features) ]
       ├──► [ Logistic Regression Classifier (76.44% Acc) ]
       ├──► [ XAI Mathematical Log-Odds Engine ]
       └──► [ SQLite Database (WAL Mode Concurrency) ]
```

---

## 2. Frontend Architecture (UI / UX)

| Technology | Role / Purpose | Key Features & Implementation Details |
| :--- | :--- | :--- |
| **HTML5** | Application Structure | Semantic, accessible layout with input panels, confidence meters, XAI cloud, and audit drawers. |
| **Jinja2** | Template Engine | Server-side template rendering injected dynamically by FastAPI (`templates/index.html`). |
| **Vanilla CSS3** | Styling & Theme | **Dark Glassmorphism Theme**: Translucent cards (`rgba(16, 24, 40, 0.75)`), backdrop blur (`16px`), custom CSS variables, responsive Grid & Flexbox, smooth transitions. |
| **Vanilla JavaScript (ES6+)** | Dynamic Interactivity | Client-side application logic (`static/js/app.js`): asynchronous `fetch` calls, dynamic DOM manipulation, real-time token highlighting, animated progress bars, debounce input listeners, and modal management. |
| **Google Fonts** | Typography | Modern, high-legibility typography using the *Outfit* and *Inter* font families. |
| **Custom SVG Badges & Icons** | Visual Indicators | Neon category-specific condition badges (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder). |

---

## 3. Backend & REST API Layer

| Technology | Role / Purpose | Key Features & Implementation Details |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Programming Language | Strong typing, modern `pathlib`, dataclasses, and high-performance libraries. |
| **FastAPI** (`0.100+`) | REST API Web Framework | High-performance, asynchronous web framework built on Starlette and Pydantic. |
| **Uvicorn** (`0.23+`) | ASGI Web Server | Production-ready, lightning-fast asynchronous server runtime (`uvicorn.run`). |
| **Pydantic v2** (`2.0+`) | Schema Validation | Type validation for incoming request bodies (`PredictionRequest`, `FeedbackRequest`) and response models. |
| **Starlette** | Core ASGI Components | Static file mounting (`StaticFiles`), Jinja2 templating, and background tasks. |

### REST Endpoints Implemented:
- `POST /api/predict` — Multi-class inference + XAI token attribution + SQLite auto-save
- `POST /api/explain` — Isolated word-level log-odds extraction
- `GET /api/history` — Paginated assessment history with category filtering
- `DELETE /api/history/{id}` — Individual record deletion
- `DELETE /api/history` — Bulk database history purge
- `POST /api/history/{id}/feedback` — Human-in-the-loop accuracy ratings & category corrections
- `GET /api/analytics` — Condition distributions & total usage metrics
- `GET /api/info` — Active model parameters, n-gram ranges, top keywords per condition
- `GET /api/examples` — Clinical test preset examples
- `GET /api/health` — System health check & database connection probe
- `GET /docs` & `GET /redoc` — Auto-generated interactive Swagger / ReDoc API documentation

---

## 4. Machine Learning & NLP Pipeline

| Technology | Role / Purpose | Key Features & Implementation Details |
| :--- | :--- | :--- |
| **Scikit-Learn** (`1.3+`) | ML Framework | Model training, cross-validation, grid search, and evaluation metrics. |
| **TfidfVectorizer** | Feature Extraction | Sublinear TF scaling, n-gram range `(1, 2)`, max features `5,000`, English stopword filtering. |
| **Logistic Regression** (Active Model) | Multi-Class Classification | Regularized L2 penalty (`C=4.28`), `lbfgs` solver, yielding **76.44% accuracy** across 7 mental health classes. |
| **Linear SVM (`LinearSVC`)** | Benchmark Classifier | Squared hinge loss (`C=0.234`), 76.93% accuracy. |
| **Random Forest (`RandomForestClassifier`)** | Benchmark Classifier | 500 decision trees, Gini impurity evaluation (75.37% accuracy). |
| **Multinomial Naive Bayes** | Benchmark Baseline | Fast probabilistic baseline model. |
| **Joblib** (`1.3+`) | Model Serialization | Efficient binary persistence of trained models (`model.joblib`), vectorizers, and label encoders. |
| **NumPy** (`1.24+`) | Vector Math | Array manipulations, logit calculations, and matrix operations. |
| **Pandas** (`2.0+`) | Data Wrangling | CSV ingestion, dataset cleaning, stratification, and dataset statistics. |
| **Regex (`re`)** | Text Sanitization | Custom preprocessor (`src/utils/text_preprocessing.py`) stripping URLs, mentions, emojis, and special noise while preserving contraction semantics. |

---

## 5. Explainable AI (XAI) Engine

| Technology / Method | Role / Purpose | Key Features & Implementation Details |
| :--- | :--- | :--- |
| **Linear Log-Odds Decomposition** | Mathematical Token Attribution | Decomposes model logits into exact word-level contributions: $\text{Logit}_c(\mathbf{x}) = \beta_{0, c} + \sum \beta_{i, c} \cdot \text{TF-IDF}(w_i)$. |
| **Directional Attribution** | Transparent Interpretability | Distinguishes between positive evidence ($\beta > 0$, highlighted green) and contradictory terms ($\beta < 0$, highlighted red). |
| **Custom Explainability Engine** | `src/models/ml_models.py` | Calculates localized token importance, normalizes scores, and returns sorted word importance arrays to the frontend. |

---

## 6. Database & Persistence Layer

| Technology | Role / Purpose | Key Features & Implementation Details |
| :--- | :--- | :--- |
| **SQLite 3** | Embedded Relational Database | Zero-configuration, serverless SQL storage (`data/mental_health_xai.db`). |
| **WAL Mode** (Write-Ahead Logging) | Concurrency & Performance | `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` for non-blocking concurrent reads and fast writes. |
| **Thread-Safe Connection Manager** | Database Handler | `src/db/database.py` with thread-local connection pooling and dictionary row factory (`sqlite3.Row`). |
| **Audit & Feedback Schema** | Persistent Storage | Stores raw text, predicted condition, multi-class probability JSON vectors, top XAI contributing tokens, and human corrections. |

---

## 7. Testing & Quality Assurance

| Technology | Role / Purpose | Key Features & Implementation Details |
| :--- | :--- | :--- |
| **Pytest** (`7.4+`) | Automated Test Runner | Comprehensive test suite with 14 automated tests spanning models, database, and API. |
| **HTTPX / TestClient** | API Integration Testing | `fastapi.testclient.TestClient` for simulating real HTTP requests to all REST routes. |
| **Pytest Configuration** | `pytest.ini` | Root-level test path discovery and warning filters. |

---

## 8. Data Processing & Benchmarking Tools

| Technology | Role / Purpose | Key Features & Implementation Details |
| :--- | :--- | :--- |
| **Matplotlib** (`3.7+`) | Visualization Library | Generates confusion matrix heatmaps, ROC comparison curves, and model comparison charts. |
| **Seaborn** (`0.12+`) | Statistical Visualization | High-quality distribution plots and top-feature bar charts (`saved_models/workbench_experiments/`). |
| **Training & Experiment Workbench** | `train_and_experiment.py` | Multi-model benchmark suite running grid search across Logistic Regression, SVM, Random Forest, and Naive Bayes. |

---

## 9. Development & DevOps Tools

| Tool | Purpose |
| :--- | :--- |
| **Git & GitHub** | Distributed version control, milestone branch management, and remote repository hosting. |
| **PowerShell** | Local Windows script execution, virtual environment management, and process control. |
| **Virtual Environment (`venv`)** | Python isolated runtime environment preventing global dependency conflicts. |
| **`.gitignore`** | Ignores cache directories (`__pycache__/`, `.pytest_cache/`) and local runtime database files. |

---

## 10. Complete Stack Summary Table

| Category | Primary Technologies Used |
| :--- | :--- |
| **Frontend** | HTML5, Jinja2, Vanilla CSS3 (Dark Glassmorphism), Vanilla JS (ES6+), Google Fonts |
| **Backend Framework** | FastAPI (Python 3.10+), Starlette, Pydantic v2 |
| **Web Server** | Uvicorn (ASGI) |
| **Machine Learning** | Scikit-Learn (TF-IDF, Logistic Regression, Linear SVM, Random Forest, Naive Bayes), Joblib |
| **Data & Scientific** | NumPy, Pandas, Regex (`re`) |
| **Explainable AI (XAI)** | Word-Level Mathematical Log-Odds Decomposition Engine |
| **Database** | SQLite 3 with Write-Ahead Logging (WAL Mode) |
| **Testing** | Pytest, FastAPI TestClient (HTTPX) |
| **Visualizations** | Matplotlib, Seaborn |
| **API Architecture** | Self-Hosted RESTful API with OpenAPI / Swagger UI |

---

## 11. Presentation & Voice-Over Talking Points

Use these short bullet points during viva, presentations, or video voice-overs:

- 💻 **Frontend**: *"A zero-framework, high-performance web dashboard built with HTML5, Jinja2, and modern Vanilla CSS3 featuring a responsive Dark Glassmorphism aesthetic and dynamic ES6 JavaScript."*
- ⚡ **Backend**: *"An asynchronous, low-latency REST API powered by FastAPI and Uvicorn with Pydantic data schemas."*
- 🧠 **ML & XAI**: *"Trained on 53,000+ clinical text samples across 7 conditions using Scikit-Learn TF-IDF and regularized Logistic Regression, paired with a custom token log-odds explainability engine."*
- 🗄️ **Database**: *"An embedded SQLite database running in WAL mode for thread-safe inference auditing and human-in-the-loop feedback persistence."*
- 🔒 **Privacy & Autonomy**: *"100% self-contained and local — no external third-party API keys, subscription fees, or cloud privacy risks."*
