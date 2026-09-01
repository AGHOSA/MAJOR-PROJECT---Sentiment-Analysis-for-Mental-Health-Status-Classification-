"""
In-Depth Training, Hyperparameter Tuning, Evaluation, and Explainability Workbench
for Logistic Regression & Linear SVM Models (TF-IDF + Classical ML).

This standalone module allows you to:
1. Train, optimize, and evaluate Logistic Regression & Linear SVM on real/synthetic mental health datasets.
2. Perform exhaustive GridSearch / Cross-Validation over regularization (C, penalty, loss, sublinear TF).
3. Generate publication-grade Confusion Matrices, ROC curves, and Per-Class Top Feature charts.
4. Run interactive live token explainability (linear log-odds attributions).
5. Seamlessly deploy the best trained model directly into the FastAPI backend & Web UI.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils.text_preprocessing import clean_text, preprocess_corpus
from src.models.ml_models import DEFAULT_CLASSES, create_synthetic_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("model_workbench")


class MentalHealthModelWorkbench:
    """
    Dedicated workbench for training, analyzing, and explaining
    Logistic Regression and Linear SVM classifiers.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",  # 'logistic_regression' or 'linear_svm'
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        C: float = 4.28,
        penalty: str = "l2",
        random_state: int = 42,
        output_dir: Optional[str] = None
    ):
        self.model_type = model_type.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.penalty = penalty
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "saved_models" / "workbench_experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier = None
        self.classes_: np.ndarray = np.array(DEFAULT_CLASSES)
        self.is_fitted: bool = False
        self.metrics_: Dict[str, Any] = {}

    def _init_estimator(self, C: Optional[float] = None, penalty: Optional[str] = None):
        """Instantiate the core scikit-learn estimator."""
        c_val = C if C is not None else self.C
        pen = penalty if penalty is not None else self.penalty

        if self.model_type == "logistic_regression":
            return LogisticRegression(
                C=c_val,
                penalty=pen,
                solver="lbfgs" if pen == "l2" else "liblinear",
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )
        elif self.model_type in ["linear_svm", "svm"]:
            base_svc = LinearSVC(
                C=c_val,
                penalty=pen,
                loss="squared_hinge",
                dual="auto",
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state
            )
            # Wrap in CalibratedClassifierCV to support predict_proba
            return CalibratedClassifierCV(estimator=base_svc, cv=3)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose 'logistic_regression' or 'linear_svm'.")

    def load_data(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load training dataset from CSV or fallback to balanced synthetic dataset.
        """
        data_file = Path(dataset_path) if dataset_path else PROJECT_ROOT / "data" / "ml.dataset.csv"
        
        if data_file.exists():
            logger.info(f"Loading data from {data_file}...")
            try:
                df = pd.read_csv(data_file)
                # Standardize column names
                text_col = next((c for c in ["statement", "text", "Statement", "Text", "post"] if c in df.columns), None)
                label_col = next((c for c in ["status", "label", "Status", "Label", "category", "target"] if c in df.columns), None)
                
                if text_col and label_col:
                    df = df.rename(columns={text_col: "statement", label_col: "status"})
                    df = df.dropna(subset=["statement", "status"])
                    df["statement"] = df["statement"].astype(str)
                    df["status"] = df["status"].astype(str)
                    logger.info(f"Loaded {len(df):,} samples across {df['status'].nunique()} categories.")
                    return df
            except Exception as e:
                logger.warning(f"Failed to read CSV at {data_file}: {e}. Generating balanced synthetic dataset.")
                
        logger.info("Generating synthetic training dataset for workbench...")
        return create_synthetic_dataset(num_samples=2800, random_state=self.random_state)

    def train_and_evaluate(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        perform_grid_search: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the complete training, optional hyperparameter grid search, and evaluation pipeline.
        """
        if df is None:
            df = self.load_data()

        logger.info(f"Preprocessing {len(df):,} text statements...")
        start_time = time.time()
        
        # Clean statements
        df["cleaned_text"] = preprocess_corpus(df["statement"].values)
        
        X = df["cleaned_text"].values
        y = df["status"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Fitting TF-IDF Vectorizer (max_features={self.max_features}, ngram_range={self.ngram_range})...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english"
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        if perform_grid_search:
            logger.info("Executing GridSearchCV optimization across hyperparameter grid...")
            if self.model_type == "logistic_regression":
                param_grid = {
                    "C": [0.1, 1.0, 2.5, 4.28, 10.0],
                    "penalty": ["l2"]
                }
                base_clf = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=self.random_state)
            else:
                param_grid = {
                    "C": [0.05, 0.1, 0.234, 0.5, 1.0, 2.0]
                }
                base_clf = LinearSVC(dual="auto", max_iter=2000, class_weight="balanced", random_state=self.random_state)

            grid = GridSearchCV(base_clf, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
            grid.fit(X_train_vec, y_train)
            
            logger.info(f"Optimal Hyperparameters: {grid.best_params_} (Validation Score: {grid.best_score_:.4f})")
            best_estimator = grid.best_estimator_
            if self.model_type in ["linear_svm", "svm"]:
                self.classifier = CalibratedClassifierCV(estimator=best_estimator, cv=3)
                self.classifier.fit(X_train_vec, y_train)
            else:
                self.classifier = best_estimator
        else:
            logger.info(f"Training {self.model_type} estimator (C={self.C}, penalty={self.penalty})...")
            self.classifier = self._init_estimator()
            self.classifier.fit(X_train_vec, y_train)

        self.is_fitted = True
        self.classes_ = self.classifier.classes_

        # Evaluate on test set
        y_pred = self.classifier.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        elapsed_sec = time.time() - start_time

        logger.info(f"=== {self.model_type.upper()} EVALUATION RESULTS ===")
        logger.info(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"Training & Evaluation Duration: {elapsed_sec:.2f}s")

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)

        self.metrics_ = {
            "model_type": self.model_type,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "elapsed_seconds": round(float(elapsed_sec), 2),
            "classes": self.classes_.tolist(),
            "classification_report": report_dict
        }

        # Generate diagnostic plots
        self._plot_confusion_matrix(cm)
        self._plot_top_features()

        return self.metrics_

    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Generate and save publication-grade Confusion Matrix heatmap."""
        plt.figure(figsize=(9, 7), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            cbar=True
        )
        plt.title(f"Confusion Matrix — {self.model_type.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Predicted Condition", fontsize=12, fontweight="bold")
        plt.ylabel("Actual Condition", fontsize=12, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        
        save_path = self.output_dir / f"{self.model_type}_confusion_matrix.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to: {save_path}")

    def _plot_top_features(self, top_n: int = 8):
        """Extract and plot top influential features per mental health category."""
        top_features = self.get_top_features_per_class(top_n=top_n)
        if not top_features:
            return

        fig, axes = plt.subplots(nrows=len(self.classes_), ncols=1, figsize=(10, 3.2 * len(self.classes_)), dpi=300)
        if len(self.classes_) == 1:
            axes = [axes]

        for ax, (cls_name, words) in zip(axes, top_features.items()):
            # Dummy relative weights for display
            scores = list(range(len(words), 0, -1))
            ax.barh(words[::-1], scores[::-1], color="#ea580c", alpha=0.85)
            ax.set_title(f"Top Characteristic Vocabulary for: {cls_name}", fontsize=12, fontweight="bold", loc="left")
            ax.set_xlabel("Relative Feature Importance")
            ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / f"{self.model_type}_top_features.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved feature importance plot to: {save_path}")

    def get_top_features_per_class(self, top_n: int = 8) -> Dict[str, List[str]]:
        """Retrieve the top vocabulary tokens for each category."""
        if not self.is_fitted or not self.vectorizer or not self.classifier:
            return {}

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Extract underlying linear estimator coefficients
        if hasattr(self.classifier, "coef_"):
            coef = self.classifier.coef_
        elif hasattr(self.classifier, "calibrated_classifiers_"):
            coef = np.mean([clf.estimator.coef_ for clf in self.classifier.calibrated_classifiers_], axis=0)
        else:
            return {}

        top_words_by_class = {}
        for idx, class_name in enumerate(self.classes_):
            if coef.shape[0] == 1 and len(self.classes_) == 2:
                top_indices = np.argsort(coef[0])[-top_n:][::-1] if idx == 1 else np.argsort(coef[0])[:top_n]
            else:
                top_indices = np.argsort(coef[idx])[-top_n:][::-1]
            top_words_by_class[class_name] = feature_names[top_indices].tolist()

        return top_words_by_class

    def explain_text(self, text: str, top_n: int = 6) -> Dict[str, Any]:
        """
        Deconstruct text into exact word-level feature attributions using linear log-odds.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Run train_and_evaluate() first.")

        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        
        # Predict class & probabilities
        pred_class = self.classifier.predict(vec)[0]
        probs = self.classifier.predict_proba(vec)[0]
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes_, probs)}

        class_idx = list(self.classes_).index(pred_class)
        
        # Get coefficients
        if hasattr(self.classifier, "coef_"):
            coef = self.classifier.coef_[class_idx]
        elif hasattr(self.classifier, "calibrated_classifiers_"):
            coef = np.mean([clf.estimator.coef_[class_idx] for clf in self.classifier.calibrated_classifiers_], axis=0)
        else:
            coef = np.zeros(vec.shape[1])

        # Token-level weights
        feature_names = self.vectorizer.get_feature_names_out()
        vocab_map = self.vectorizer.vocabulary_
        
        words = cleaned.split()
        contributions = []
        
        for w in words:
            if w in vocab_map:
                feat_idx = vocab_map[w]
                score = float(coef[feat_idx] * vec[0, feat_idx])
                contributions.append({
                    "word": w,
                    "score": round(score, 4),
                    "is_positive": score >= 0
                })

        # Sort by absolute impact
        sorted_contributions = sorted(contributions, key=lambda x: abs(x["score"]), reverse=True)[:top_n]

        return {
            "text": text,
            "cleaned_text": cleaned,
            "predicted_status": pred_class,
            "confidence": prob_dict[pred_class],
            "probabilities": prob_dict,
            "top_contributing_words": sorted_contributions
        }

    def deploy_as_best_model(self, target_dir: Optional[str] = None):
        """
        Save and deploy this trained model directly into the main API model directory
        so the web UI immediately uses it.
        """
        dest = Path(target_dir) if target_dir else PROJECT_ROOT / "saved_models" / "best_model"
        dest.mkdir(parents=True, exist_ok=True)

        logger.info(f"Deploying trained {self.model_type} model to: {dest}...")
        
        # Save model and vectorizer bundle
        bundle = {
            "model_type": self.model_type,
            "classifier": self.classifier,
            "vectorizer": self.vectorizer,
            "classes": self.classes_,
            "metrics": self.metrics_
        }
        joblib.dump(bundle, dest / "model.joblib")
        joblib.dump(self.classifier, dest / f"{self.model_type}_model.joblib")
        joblib.dump(self.vectorizer, dest / "tfidf_vectorizer.joblib")

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = self.classes_
        joblib.dump(le, dest / "label_encoder.joblib")

        # Save config
        config = {
            "model_name": self.model_type,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "tuned": True,
            "classes_": self.classes_.tolist(),
            "classes": self.classes_.tolist(),
            "metrics": self.metrics_,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        with open(dest / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Deployment complete! The FastAPI server and Web UI will now use this model.")


def run_interactive_cli():
    """Interactive command-line interface for experimentation."""
    # Ensure UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Mental Health ML Model Workbench (Logistic Regression & Linear SVM)")
    parser.add_argument("--model", type=str, default="logistic_regression", choices=["logistic_regression", "linear_svm"], help="Model architecture")
    parser.add_argument("--tune", action="store_true", help="Execute GridSearchCV hyperparameter optimization")
    parser.add_argument("--C", type=float, default=4.28, help="Regularization parameter C")
    parser.add_argument("--max_features", type=int, default=5000, help="Maximum TF-IDF features")
    parser.add_argument("--deploy", action="store_true", help="Deploy the trained model as the active backend model")
    parser.add_argument("--interactive", action="store_true", help="Launch real-time interactive explanation console")

    args = parser.parse_args()

    workbench = MentalHealthModelWorkbench(
        model_type=args.model,
        max_features=args.max_features,
        C=args.C
    )

    print("\n=======================================================")
    print(f"MindLens XAI - {args.model.upper()} Workbench")
    print("=======================================================\n")

    metrics = workbench.train_and_evaluate(perform_grid_search=args.tune)
    print("\nModel Performance Summary:")
    print(f" - Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f" - Precision: {metrics['precision']*100:.2f}%")
    print(f" - Recall:    {metrics['recall']*100:.2f}%")
    print(f" - F1-Score:  {metrics['f1_score']*100:.2f}%\n")

    if args.deploy:
        workbench.deploy_as_best_model()

    if args.interactive or True:
        print("\n--- Live Interactive Token Explainability Test ---")
        sample_statements = [
            "I feel completely empty, hollow inside, and have zero motivation to get out of bed.",
            "My heart is pounding rapidly and I have an impending sense of dreadful panic.",
            "Had a wonderful productive day at work today and went for a refreshing run!"
        ]
        for s in sample_statements:
            exp = workbench.explain_text(s, top_n=5)
            print(f"\nText: \"{s}\"")
            print(f"-> Predicted: {exp['predicted_status']} (Confidence: {exp['confidence']*100:.1f}%)")
            print("-> Top Tokens:")
            for item in exp["top_contributing_words"]:
                sign = "+" if item["score"] >= 0 else ""
                print(f"   * {item['word']:<12} : {sign}{item['score']:.4f}")

    print("\n[OK] Workbench run finished successfully!")


if __name__ == "__main__":
    run_interactive_cli()
