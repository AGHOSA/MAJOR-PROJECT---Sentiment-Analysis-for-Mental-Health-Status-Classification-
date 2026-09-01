"""
SQLite Database Layer for Explainable AI Mental Health Text Classifier.

Provides thread-safe storage, retrieval, feedback tracking, and aggregated analytics
for mental health text classification inferences and token-level explainability weights.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default database location
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mental_health_xai.db"

# Thread-local storage for SQLite connections
_local = threading.local()


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Get or create a thread-safe connection to the SQLite database.
    Enables WAL mode and dictionary-style row access.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(target_path), timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency and fast reads/writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
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
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
            ON predictions(created_at DESC);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_category 
            ON predictions(predicted_category);
        """)


def save_prediction(
    statement_text: str,
    predicted_category: str,
    confidence: float,
    probabilities: Dict[str, float],
    top_contributing_words: List[Dict[str, Any]],
    model_name: str,
    top_n_requested: int = 6,
    db_path: Optional[str] = None
) -> int:
    """
    Save an inference result and word explainability tokens to the database.
    Returns the newly inserted record ID.
    """
    conn = get_db_connection(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                statement_text,
                predicted_category,
                confidence,
                probabilities_json,
                top_contributing_words_json,
                model_name,
                top_n_requested,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                statement_text.strip(),
                predicted_category,
                float(confidence),
                json.dumps(probabilities),
                json.dumps(top_contributing_words),
                model_name,
                top_n_requested,
                now_iso
            )
        )
        return cursor.lastrowid


def _parse_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Helper to convert sqlite3.Row into serializable dictionary with parsed JSONs."""
    d = dict(row)
    if "probabilities_json" in d and isinstance(d["probabilities_json"], str):
        try:
            d["probabilities"] = json.loads(d.pop("probabilities_json"))
        except Exception:
            d["probabilities"] = {}
            
    if "top_contributing_words_json" in d and isinstance(d["top_contributing_words_json"], str):
        try:
            d["top_contributing_words"] = json.loads(d.pop("top_contributing_words_json"))
        except Exception:
            d["top_contributing_words"] = []
            
    return d


def get_history(
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None,
    search: Optional[str] = None,
    db_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Retrieve paginated prediction history with optional filters.
    Returns (records, total_count).
    """
    conn = get_db_connection(db_path)
    conditions = []
    params: List[Any] = []
    
    if category:
        conditions.append("predicted_category = ?")
        params.append(category)
        
    if search:
        conditions.append("statement_text LIKE ?")
        params.append(f"%{search}%")
        
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    # Get total count
    count_cursor = conn.execute(
        f"SELECT COUNT(*) FROM predictions {where_clause}",
        params
    )
    total_count = count_cursor.fetchone()[0]
    
    # Get records
    query = f"""
        SELECT * FROM predictions 
        {where_clause} 
        ORDER BY id DESC 
        LIMIT ? OFFSET ?
    """
    fetch_params = params + [limit, offset]
    cursor = conn.execute(query, fetch_params)
    rows = cursor.fetchall()
    
    records = [_parse_row(r) for r in rows]
    return records, total_count


def get_prediction_by_id(prediction_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single prediction by its ID."""
    conn = get_db_connection(db_path)
    cursor = conn.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (prediction_id,)
    )
    row = cursor.fetchone()
    return _parse_row(row) if row else None


def update_feedback(
    prediction_id: int,
    user_feedback: str,
    feedback_notes: Optional[str] = None,
    corrected_category: Optional[str] = None,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Record user feedback / correction on a prediction."""
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute(
            """
            UPDATE predictions 
            SET user_feedback = ?, 
                feedback_notes = ?, 
                corrected_category = ?
            WHERE id = ?
            """,
            (user_feedback, feedback_notes, corrected_category, prediction_id)
        )
        if cursor.rowcount == 0:
            return None
            
    return get_prediction_by_id(prediction_id, db_path)


def delete_prediction(prediction_id: int, db_path: Optional[str] = None) -> bool:
    """Delete a single prediction from history."""
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute(
            "DELETE FROM predictions WHERE id = ?",
            (prediction_id,)
        )
        return cursor.rowcount > 0


def clear_all_history(db_path: Optional[str] = None) -> int:
    """Delete all prediction history records."""
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute("DELETE FROM predictions")
        return cursor.rowcount


def get_analytics_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculate aggregated analytics across all stored predictions."""
    conn = get_db_connection(db_path)
    
    # Total count and average confidence
    stats_cursor = conn.execute("""
        SELECT 
            COUNT(*) as total_predictions,
            AVG(confidence) as avg_confidence,
            MIN(confidence) as min_confidence,
            MAX(confidence) as max_confidence
        FROM predictions
    """)
    stats_row = stats_cursor.fetchone()
    total_predictions = stats_row["total_predictions"] or 0
    avg_confidence = round(stats_row["avg_confidence"] or 0.0, 4)
    
    # Breakdown by category
    category_cursor = conn.execute("""
        SELECT predicted_category, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM predictions
        GROUP BY predicted_category
        ORDER BY count DESC
    """)
    category_counts = {
        row["predicted_category"]: {
            "count": row["count"],
            "avg_confidence": round(row["avg_conf"], 4)
        }
        for row in category_cursor.fetchall()
    }
    
    # Feedback counts
    feedback_cursor = conn.execute("""
        SELECT user_feedback, COUNT(*) as count
        FROM predictions
        WHERE user_feedback IS NOT NULL
        GROUP BY user_feedback
    """)
    feedback_counts = {
        row["user_feedback"]: row["count"]
        for row in feedback_cursor.fetchall()
    }
    
    return {
        "total_predictions": total_predictions,
        "average_confidence": avg_confidence,
        "category_distribution": category_counts,
        "feedback_distribution": feedback_counts
    }
