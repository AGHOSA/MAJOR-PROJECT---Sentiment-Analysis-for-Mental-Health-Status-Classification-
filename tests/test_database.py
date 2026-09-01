"""
Unit tests for SQLite database layer (src/db/database.py).
"""

import os
import tempfile
import pytest
from src.db.database import (
    init_db,
    save_prediction,
    get_history,
    get_prediction_by_id,
    update_feedback,
    delete_prediction,
    clear_all_history,
    get_analytics_summary,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass


def test_init_and_save_prediction(temp_db):
    """Test saving a prediction record and retrieving its ID."""
    row_id = save_prediction(
        statement_text="I am feeling overwhelmed with anxiety.",
        predicted_category="Anxiety",
        confidence=0.925,
        probabilities={"Anxiety": 0.925, "Normal": 0.075},
        top_contributing_words=[{"word": "anxiety", "score": 1.45, "is_positive": True}],
        model_name="logistic_regression",
        top_n_requested=5,
        db_path=temp_db
    )
    assert row_id is not None
    assert row_id >= 1
    
    # Retrieve by ID
    record = get_prediction_by_id(row_id, db_path=temp_db)
    assert record is not None
    assert record["id"] == row_id
    assert record["statement_text"] == "I am feeling overwhelmed with anxiety."
    assert record["predicted_category"] == "Anxiety"
    assert record["confidence"] == 0.925
    assert record["probabilities"]["Anxiety"] == 0.925
    assert len(record["top_contributing_words"]) == 1
    assert record["top_contributing_words"][0]["word"] == "anxiety"


def test_get_history_and_filtering(temp_db):
    """Test history querying, pagination, and category filtering."""
    # Save multiple records
    save_prediction("Feeling great today!", "Normal", 0.99, {"Normal": 0.99}, [], "model", db_path=temp_db)
    save_prediction("Very sad and empty", "Depression", 0.88, {"Depression": 0.88}, [], "model", db_path=temp_db)
    save_prediction("Cannot sleep at all", "Depression", 0.85, {"Depression": 0.85}, [], "model", db_path=temp_db)
    
    # Get all history
    history, total = get_history(limit=10, offset=0, db_path=temp_db)
    assert total == 3
    assert len(history) == 3
    
    # Filter by category
    dep_history, dep_total = get_history(category="Depression", db_path=temp_db)
    assert dep_total == 2
    assert len(dep_history) == 2
    assert all(r["predicted_category"] == "Depression" for r in dep_history)
    
    # Search by keyword
    search_history, search_total = get_history(search="great", db_path=temp_db)
    assert search_total == 1
    assert search_history[0]["predicted_category"] == "Normal"


def test_feedback_and_analytics(temp_db):
    """Test user feedback tracking and analytics generation."""
    row_id = save_prediction("Feeling stressed out", "Stress", 0.80, {"Stress": 0.80}, [], "model", db_path=temp_db)
    
    # Update feedback
    updated = update_feedback(row_id, user_feedback="accurate", feedback_notes="Spot on!", db_path=temp_db)
    assert updated is not None
    assert updated["user_feedback"] == "accurate"
    assert updated["feedback_notes"] == "Spot on!"
    
    # Analytics summary
    analytics = get_analytics_summary(db_path=temp_db)
    assert analytics["total_predictions"] == 1
    assert analytics["average_confidence"] == 0.80
    assert "Stress" in analytics["category_distribution"]
    assert analytics["category_distribution"]["Stress"]["count"] == 1
    assert analytics["feedback_distribution"].get("accurate") == 1


def test_delete_and_clear_history(temp_db):
    """Test single record deletion and clearing all history."""
    id1 = save_prediction("Text 1", "Normal", 0.9, {}, [], "model", db_path=temp_db)
    id2 = save_prediction("Text 2", "Anxiety", 0.9, {}, [], "model", db_path=temp_db)
    
    # Delete single
    deleted = delete_prediction(id1, db_path=temp_db)
    assert deleted is True
    assert get_prediction_by_id(id1, db_path=temp_db) is None
    
    # Clear all
    cleared = clear_all_history(db_path=temp_db)
    assert cleared == 1
    
    history, total = get_history(db_path=temp_db)
    assert total == 0
    assert len(history) == 0
