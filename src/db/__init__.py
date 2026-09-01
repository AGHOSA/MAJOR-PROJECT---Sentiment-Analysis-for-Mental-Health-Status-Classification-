"""
Database module for mental health XAI application.
"""

from src.db.database import (
    DEFAULT_DB_PATH,
    init_db,
    get_db_connection,
    save_prediction,
    get_history,
    get_prediction_by_id,
    update_feedback,
    delete_prediction,
    clear_all_history,
    get_analytics_summary,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "init_db",
    "get_db_connection",
    "save_prediction",
    "get_history",
    "get_prediction_by_id",
    "update_feedback",
    "delete_prediction",
    "clear_all_history",
    "get_analytics_summary",
]
