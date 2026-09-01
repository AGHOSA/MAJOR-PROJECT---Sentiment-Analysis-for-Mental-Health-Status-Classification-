"""
Quick Interactive Python Script for In-Depth Experimentation with
Logistic Regression & Linear SVM Models in MindLens XAI.

Run this script directly or import it into your interactive environment/notebook.
"""

import sys
from train_and_experiment import MentalHealthModelWorkbench

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("==================================================================")
    print("MindLens XAI - In-Depth Model Experimentation Lab")
    print("==================================================================")

    # 1. Initialize Workbench for Logistic Regression
    print("\n[Step 1] Initializing & Training Tuned Logistic Regression...")
    lr_bench = MentalHealthModelWorkbench(
        model_type="logistic_regression",
        max_features=5000,
        ngram_range=(1, 2),
        C=4.28
    )
    lr_metrics = lr_bench.train_and_evaluate()
    print(f" -> Logistic Regression Accuracy: {lr_metrics['accuracy']*100:.2f}% | F1: {lr_metrics['f1_score']:.4f}")

    # 2. Initialize Workbench for Linear SVM
    print("\n[Step 2] Initializing & Training Calibrated Linear SVM...")
    svm_bench = MentalHealthModelWorkbench(
        model_type="linear_svm",
        max_features=5000,
        ngram_range=(1, 2),
        C=0.234
    )
    svm_metrics = svm_bench.train_and_evaluate()
    print(f" -> Linear SVM Accuracy: {svm_metrics['accuracy']*100:.2f}% | F1: {svm_metrics['f1_score']:.4f}")

    # 3. Custom Text Explainability Comparison
    print("\n[Step 3] Comparing Token-Level Explainability on Custom Statement:")
    custom_statement = "Burnt out from working 80 hours a week with impossible deadlines and panic attacks."
    print(f"\nEvaluating: \"{custom_statement}\"")

    lr_exp = lr_bench.explain_text(custom_statement, top_n=5)
    print(f"\n[Logistic Regression Result]: {lr_exp['predicted_status']} ({lr_exp['confidence']*100:.1f}%)")
    for item in lr_exp["top_contributing_words"]:
        print(f"   * {item['word']:<15} -> Score: {item['score']:+.4f}")

    svm_exp = svm_bench.explain_text(custom_statement, top_n=5)
    print(f"\n[Linear SVM Result]: {svm_exp['predicted_status']} ({svm_exp['confidence']*100:.1f}%)")
    for item in svm_exp["top_contributing_words"]:
        print(f"   * {item['word']:<15} -> Score: {item['score']:+.4f}")

    # 4. Deploy Best Model to Web App
    print("\n[Step 4] Deploying Model to FastAPI Web Application...")
    lr_bench.deploy_as_best_model()

    print("\n[OK] All experiments executed successfully! Check saved_models/workbench_experiments/ for high-res PNG plots.")

if __name__ == "__main__":
    main()
