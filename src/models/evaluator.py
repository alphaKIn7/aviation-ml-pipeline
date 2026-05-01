# ──────────────────────────────────────────────
# evaluator.py — Model Evaluation
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Compute metrics, generate classification/regression reports,
#   and create visualizations (confusion matrix, ROC curve, etc.)
#   to judge model performance.
#
# WHY THIS IS CRITICAL:
#   A model that scores 99% accuracy on training data means NOTHING
#   without proper evaluation. This module ensures you're measuring
#   the right things in the right way.
#
# FUNCTIONS TO BUILD HERE:
#   - compute_metrics(y_true, y_pred, task_type)       → Dict of metrics
#   - classification_report(y_true, y_pred)             → Precision/Recall/F1
#   - plot_confusion_matrix(y_true, y_pred, save_path)  → Heatmap
#   - plot_roc_curve(y_true, y_proba, save_path)        → ROC + AUC
#   - plot_feature_importance(model, names, save_path)  → Bar chart
#   - generate_evaluation_report(results, save_path)    → Full HTML/PDF report
#
# INDUSTRY PATTERN:
#   Always log metrics to a tracking tool (MLflow, W&B). This module
#   should return metrics as dictionaries — the pipeline layer decides
#   where to log them.
# ──────────────────────────────────────────────
