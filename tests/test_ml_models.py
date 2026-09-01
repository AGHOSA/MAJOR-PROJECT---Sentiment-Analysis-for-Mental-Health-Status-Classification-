"""
Unit tests for Machine Learning Baseline Models (TF-IDF + ML).
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.utils.text_preprocessing import clean_text, preprocess_corpus
from src.models.ml_models import (
    MentalHealthMLClassifier,
    create_synthetic_dataset,
    load_mental_health_data,
    train_and_evaluate_all_models
)


def test_clean_text():
    raw_text = "Check out https://example.com/test @user I can't sleep &amp; feel exhausted!!!"
    cleaned = clean_text(raw_text)
    assert "http" not in cleaned
    assert "@user" not in cleaned
    assert "cannot" in cleaned
    assert "exhausted" in cleaned
    assert "&amp;" not in cleaned


def test_preprocess_corpus():
    corpus = ["Hello world!", "Feeling very down today..."]
    cleaned = preprocess_corpus(corpus)
    assert len(cleaned) == 2
    assert cleaned[0] == "hello world"
    assert cleaned[1] == "feeling very down today"


def test_synthetic_dataset_generation():
    df = create_synthetic_dataset(num_samples=140, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert "statement" in df.columns
    assert "status" in df.columns
    assert df["status"].nunique() == 7
    assert len(df) >= 70


def test_ml_classifier_fit_predict_save_load():
    df = create_synthetic_dataset(num_samples=210, random_state=42)
    X = df["statement"].values
    y = df["status"].values
    
    clf = MentalHealthMLClassifier(model_name="logistic_regression", max_features=500, tuned=True)
    clf.fit(X[:150], y[:150])
    
    assert clf.is_fitted
    preds = clf.predict(X[150:])
    assert len(preds) == len(X[150:])
    assert all(isinstance(p, str) for p in preds)
    
    probs = clf.predict_proba(X[150:])
    assert probs.shape == (len(X[150:]), len(clf.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    # Test local explanation
    exp = clf.explain_prediction("I am experiencing severe panic attacks and anxiety every single day.")
    assert "predicted_status" in exp
    assert "probabilities" in exp
    assert "top_contributing_words" in exp
    assert len(exp["top_contributing_words"]) > 0
    
    # Test save & load
    temp_dir = tempfile.mkdtemp()
    try:
        clf.save(temp_dir)
        loaded_clf = MentalHealthMLClassifier.load(temp_dir)
        assert loaded_clf.is_fitted
        loaded_preds = loaded_clf.predict(X[150:])
        assert loaded_preds == preds
    finally:
        shutil.rmtree(temp_dir)


def test_evaluate_and_compare_all_models():
    df = create_synthetic_dataset(num_samples=280, random_state=42)
    temp_dir = tempfile.mkdtemp()
    try:
        results, best_model = train_and_evaluate_all_models(
            df=df,
            test_size=0.25,
            tuned=False,
            output_dir=temp_dir
        )
        assert "logistic_regression" in results
        assert "linear_svm" in results
        assert "random_forest" in results
        assert "naive_bayes" in results
        assert best_model in results
        assert 0.0 <= results[best_model]["accuracy"] <= 1.0
        assert os.path.exists(os.path.join(temp_dir, "best_model"))
        assert os.path.exists(os.path.join(temp_dir, "model_comparison_summary.json"))
        assert os.path.exists(os.path.join(temp_dir, "model_comparison.png"))
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
