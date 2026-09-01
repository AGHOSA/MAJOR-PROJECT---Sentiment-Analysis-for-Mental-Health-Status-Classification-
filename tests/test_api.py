"""
Unit tests for FastAPI endpoints in api/main.py including SQLite database persistence.
"""

import pytest
import asyncio
from starlette.requests import Request
from api.main import (
    app,
    serve_ui,
    startup_event,
    health_check,
    get_model_info,
    get_preset_examples,
    predict_mental_health,
    fetch_history,
    fetch_prediction_by_id,
    submit_prediction_feedback,
    remove_history_record,
    clear_history,
    fetch_analytics,
    PredictionRequest,
    FeedbackRequest,
)


def setup_module(module):
    """Ensure DB and classifier model are initialized before running API tests."""
    startup_event()


def test_serve_ui():
    """Verify that the home page template renders successfully."""
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    resp = asyncio.run(serve_ui(req))
    assert resp.status_code == 200
    assert b"MindLens" in resp.body


def test_health_check():
    """Verify the health check endpoint returns DB status."""
    resp = asyncio.run(health_check())
    assert resp["status"] == "healthy"
    assert resp["model_loaded"] is True
    assert "database" in resp
    assert resp["classes_count"] == 7


def test_get_preset_examples():
    """Verify the preset examples endpoint."""
    resp = asyncio.run(get_preset_examples())
    assert "examples" in resp
    assert len(resp["examples"]) > 0


def test_predict_and_db_persistence():
    """Verify /predict classifies text, returns explainability, and saves to database."""
    payload = PredictionRequest(
        text="I feel completely drained, hopeless, and exhausted every single day.",
        top_n_words=5
    )
    resp = asyncio.run(predict_mental_health(payload))
    assert resp.predicted_category in ["Depression", "Anxiety", "Stress", "Normal", "Bipolar", "Personality disorder", "Suicidal"]
    assert 0.0 <= resp.confidence <= 1.0
    assert len(resp.top_contributing_words) > 0
    assert resp.id is not None
    assert resp.id >= 1
    
    # Verify history endpoint returns this prediction
    history_res = asyncio.run(fetch_history(limit=10, offset=0, category=None, search=None))
    assert history_res["total"] >= 1
    assert any(item["id"] == resp.id for item in history_res["items"])
    
    # Verify single prediction retrieval
    single_record = asyncio.run(fetch_prediction_by_id(resp.id))
    assert single_record["id"] == resp.id
    assert single_record["statement_text"] == payload.text
    
    # Submit feedback
    feedback_res = asyncio.run(submit_prediction_feedback(
        resp.id,
        FeedbackRequest(user_feedback="accurate", feedback_notes="Verified by test")
    ))
    assert feedback_res["message"] == "Feedback recorded successfully"
    assert feedback_res["record"]["user_feedback"] == "accurate"
    
    # Check analytics endpoint
    analytics = asyncio.run(fetch_analytics())
    assert analytics["total_predictions"] >= 1
    assert analytics["average_confidence"] > 0.0
    
    # Delete single record
    delete_res = asyncio.run(remove_history_record(resp.id))
    assert delete_res["message"] == "Record deleted successfully"


def test_clear_all_history_endpoint():
    """Verify clearing all history from database."""
    payload = PredictionRequest(text="Another test statement to clear later.", top_n_words=3)
    asyncio.run(predict_mental_health(payload))
    
    res = asyncio.run(clear_history())
    assert "deleted_count" in res
    assert res["deleted_count"] >= 1
