"""
FastAPI Backend for Explainable AI Mental Health Text Classification.

Provides REST endpoints for inference, class probability estimations,
word-level explainability feature attributions, persistent SQLite history,
and user feedback tracking using the trained TF-IDF + ML pipeline.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.models.ml_models import MentalHealthMLClassifier, create_synthetic_dataset
from src.db import (
    init_db,
    save_prediction,
    get_history,
    get_prediction_by_id,
    update_feedback,
    delete_prediction,
    clear_all_history,
    get_analytics_summary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("api")

# Directory paths
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
BEST_MODEL_DIR = SAVED_MODELS_DIR / "best_model"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Global pipeline instance
classifier: Optional[MentalHealthMLClassifier] = None
model_metadata: Dict[str, Any] = {}

# Example statements for UI testing
PRESET_EXAMPLES = [
    {
        "category": "Normal",
        "label": "Positive / Normal",
        "text": "Had a wonderfully productive day at work today and went for a refreshing run in the park!"
    },
    {
        "category": "Depression",
        "label": "Depression",
        "text": "I feel completely empty, hollow inside, and have zero motivation to get out of bed. The sadness is overwhelming."
    },
    {
        "category": "Suicidal",
        "label": "Suicidal Ideation",
        "text": "I cannot endure this unbearable agony anymore, I feel like everyone would be much better off without my existence."
    },
    {
        "category": "Anxiety",
        "label": "Anxiety",
        "text": "My heart is pounding rapidly, I can barely breathe, and I have a constant impending dread about everything."
    },
    {
        "category": "Bipolar",
        "label": "Bipolar Condition",
        "text": "Last week I was on top of the world with manic euphoria, but today I crashed into devastating melancholic depression."
    },
    {
        "category": "Stress",
        "label": "Severe Stress / Burnout",
        "text": "Burnt out from working 80 hours a week with impossible deadlines, unpaid bills, and non-stop pressure."
    },
    {
        "category": "Personality disorder",
        "label": "Personality Disorder",
        "text": "My fear of abandonment causes me to push everyone away, experiencing extreme emotional dysregulation and identity confusion."
    }
]


def load_model_pipeline() -> MentalHealthMLClassifier:
    """Load or initialize the ML classification pipeline."""
    global classifier, model_metadata
    
    if BEST_MODEL_DIR.exists() and (BEST_MODEL_DIR / "config.json").exists():
        logger.info(f"Loading trained model from {BEST_MODEL_DIR}...")
        classifier = MentalHealthMLClassifier.load(str(BEST_MODEL_DIR))
    else:
        logger.warning("No pre-trained model found at saved_models/best_model. Training a baseline pipeline now...")
        df = create_synthetic_dataset(num_samples=2100)
        classifier = MentalHealthMLClassifier(model_name="logistic_regression", max_features=5000, tuned=True)
        classifier.fit(df["statement"].values, df["status"].values)
        classifier.save(str(BEST_MODEL_DIR))
        
    # Load summary metrics if available
    summary_file = SAVED_MODELS_DIR / "model_comparison_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics summary: {e}")
            
    return classifier


# Initialize FastAPI App
app = FastAPI(
    title="Explainable AI Mental Health Classification API",
    description="REST API for predicting mental health categories, extracting token explainability weights, persisting assessment history, and user feedback.",
    version="1.1.0"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and templates folders
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
def startup_event():
    """Load ML pipeline and initialize SQLite database on application startup."""
    init_db()
    load_model_pipeline()
    logger.info(f"Database initialized and Mental Health Classifier loaded. Active classes: {classifier.classes_.tolist() if classifier else []}")


# --- Pydantic Schemas ---

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        description="The statement or social media post to analyze."
    )
    top_n_words: Optional[int] = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of top influential words to return for explainability."
    )


class WordContribution(BaseModel):
    word: str
    score: float
    is_positive: bool


class PredictionResponse(BaseModel):
    id: Optional[int] = None
    text: str
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    top_contributing_words: List[WordContribution]
    model_name: str
    timestamp: str


class FeedbackRequest(BaseModel):
    user_feedback: str = Field(..., description="'accurate', 'inaccurate', 'helpful', etc.")
    feedback_notes: Optional[str] = Field(default=None, description="Optional notes or context.")
    corrected_category: Optional[str] = Field(default=None, description="Correct category if prediction was incorrect.")


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the interactive web interface."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_name": classifier.model_name.replace("_", " ").title() if classifier else "TF-IDF + ML",
            "classes": classifier.classes_.tolist() if classifier else [],
            "examples": PRESET_EXAMPLES
        }
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None and classifier.is_fitted,
        "active_model": classifier.model_name if classifier else None,
        "classes_count": len(classifier.classes_) if classifier else 0,
        "database": "connected (sqlite3)",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/info")
async def get_model_info():
    """Retrieve metadata, available classes, and model comparison metrics."""
    if not classifier:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    return {
        "model_name": classifier.model_name,
        "max_features": classifier.max_features,
        "ngram_range": classifier.ngram_range,
        "classes": classifier.classes_.tolist(),
        "top_class_features": classifier.get_top_features_per_class(top_n=5),
        "comparison_metrics": model_metadata
    }


@app.get("/api/examples")
async def get_preset_examples():
    """Return preset example statements for demonstration."""
    return {"examples": PRESET_EXAMPLES}


@app.post("/predict", response_model=PredictionResponse)
async def predict_mental_health(payload: PredictionRequest):
    """
    Classify input text into one of 7 mental health categories, return word-level attributions,
    and persist the prediction result into SQLite database.
    """
    global classifier
    if not classifier or not classifier.is_fitted:
        load_model_pipeline()
        
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        # Run explain_prediction
        explanation = classifier.explain_prediction(text, top_n=payload.top_n_words)
        predicted_status = explanation["predicted_status"]
        probabilities = explanation.get("probabilities", {})
        
        confidence = probabilities.get(predicted_status, 1.0)
        
        raw_contributions = explanation.get("top_contributing_words", [])
        contributions = [
            WordContribution(
                word=w,
                score=round(s, 4),
                is_positive=(s >= 0)
            )
            for w, s in raw_contributions
        ]
        
        # Save to SQLite Database
        saved_id = None
        try:
            saved_id = save_prediction(
                statement_text=text,
                predicted_category=predicted_status,
                confidence=round(float(confidence), 4),
                probabilities={k: round(float(v), 4) for k, v in probabilities.items()},
                top_contributing_words=[c.model_dump() for c in contributions],
                model_name=classifier.model_name,
                top_n_requested=payload.top_n_words
            )
        except Exception as db_err:
            logger.warning(f"Failed to persist prediction to database: {db_err}")
        
        return PredictionResponse(
            id=saved_id,
            text=text,
            predicted_category=predicted_status,
            confidence=round(float(confidence), 4),
            probabilities={k: round(float(v), 4) for k, v in probabilities.items()},
            top_contributing_words=contributions,
            model_name=classifier.model_name,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction failed for text: {text}. Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


# --- Database & History Endpoints ---

@app.get("/api/history")
async def fetch_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    category: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None)
):
    """Retrieve paginated assessment history stored in SQLite database."""
    records, total_count = get_history(limit=limit, offset=offset, category=category, search=search)
    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "items": records
    }


@app.get("/api/history/{prediction_id}")
async def fetch_prediction_by_id(prediction_id: int):
    """Fetch complete details of a single historical assessment."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return record


@app.delete("/api/history/{prediction_id}")
async def remove_history_record(prediction_id: int):
    """Delete a single assessment from database history."""
    success = delete_prediction(prediction_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found or already deleted.")
    return {"message": "Record deleted successfully", "id": prediction_id}


@app.delete("/api/history")
async def clear_history():
    """Clear all assessment history from database."""
    deleted_count = clear_all_history()
    return {"message": "All history records cleared", "deleted_count": deleted_count}


@app.post("/api/history/{prediction_id}/feedback")
async def submit_prediction_feedback(prediction_id: int, payload: FeedbackRequest):
    """Submit user feedback / correction on a prediction."""
    updated = update_feedback(
        prediction_id=prediction_id,
        user_feedback=payload.user_feedback,
        feedback_notes=payload.feedback_notes,
        corrected_category=payload.corrected_category
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return {"message": "Feedback recorded successfully", "record": updated}


@app.get("/api/analytics")
async def fetch_analytics():
    """Retrieve aggregate statistics and category distribution from database."""
    return get_analytics_summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
