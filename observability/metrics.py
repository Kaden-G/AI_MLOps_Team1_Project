"""
Evaluation Metrics Module

ROLE IN OBSERVABILITY LAYER:
-----------------------------
This module provides comprehensive, clinical-grade evaluation metrics for binary
classification of brain tumor detection. It directly supports:
- MLO 8.6: Rigorous ML performance evaluation
- MLO 14.1: Metrics visualization for final demo
- Project SMART goals: Achieving >=80% precision and recall

WHY THESE METRICS?
------------------
For brain tumor detection, different metrics have different clinical implications:

1. RECALL (Sensitivity): Most critical!
   - Measures: "Of all actual tumors, how many did we catch?"
   - Low recall = MISSED TUMORS = dangerous!
   - Target: >= 0.80 (from project proposal)
   - Clinical impact: False negatives can be fatal

2. PRECISION (Positive Predictive Value):
   - Measures: "Of all tumor predictions, how many were correct?"
   - Low precision = unnecessary scans, anxiety, procedures
   - Target: >= 0.80 (from project proposal)
   - Clinical impact: False positives waste resources, cause stress

3. F1 SCORE:
   - Harmonic mean of precision and recall
   - Balances both concerns
   - Good single metric for model comparison

4. ACCURACY:
   - Overall correctness
   - Can be misleading if classes are imbalanced!
   - Example: 95% "no tumor" cases -> always predicting "no tumor" gives 95% accuracy but 0% recall!

5. ROC-AUC:
   - Measures discrimination ability across all thresholds
   - Threshold-independent view of model quality
   - Useful for comparing models

ARCHITECTURE-AGNOSTIC DESIGN:
------------------------------
All functions accept only:
- y_true: Ground truth binary labels (0 or 1)
- y_pred: Predicted binary labels (0 or 1)
- y_prob: Prediction probabilities (optional, for ROC-AUC)

No assumptions about model architecture, framework, or implementation!
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =============================================================================
# CORE METRICS COMPUTATION
# =============================================================================

def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    positive_label: int = 1
) -> Dict[str, float]:
    """
    Compute all standard classification metrics for binary classification.

    This is the main function for evaluating model performance. It computes
    all metrics needed for the project in one call.

    Args:
        y_true (array-like): Ground truth binary labels.
            Shape: (n_samples,), values: 0 (no tumor) or 1 (tumor).
        y_pred (array-like): Predicted binary labels.
            Shape: (n_samples,), values: 0 or 1.
        y_prob (array-like, optional): Prediction probabilities for positive class.
            Shape: (n_samples,) or (n_samples, 2), values in [0, 1].
            If shape is (n_samples, 2), uses second column (positive class prob).
            Required for ROC-AUC calculation.
        positive_label (int): Which label is considered "positive" (default: 1 = tumor).

    Returns:
        dict: Dictionary containing:
            - 'accuracy': Overall accuracy
            - 'precision': Precision (PPV) for tumor class
            - 'recall': Recall (sensitivity) for tumor class - MOST CRITICAL!
            - 'f1': F1 score (harmonic mean of precision and recall)
            - 'specificity': True negative rate (complement of false positive rate)
            - 'roc_auc': Area under ROC curve (if y_prob provided)
            - 'support_positive': Number of actual positive cases
            - 'support_negative': Number of actual negative cases

    Example:
        >>> # After training/inference
        >>> y_true = [0, 1, 1, 0, 1, 0, 1, 1]  # Ground truth
        >>> y_pred = [0, 1, 1, 0, 0, 0, 1, 1]  # Model predictions
        >>> y_prob = [0.1, 0.9, 0.85, 0.2, 0.45, 0.15, 0.95, 0.88]  # Probabilities
        >>>
        >>> metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        >>> print(f"Recall: {metrics['recall']:.2%}")  # e.g., "Recall: 80.00%"
        >>> print(f"Precision: {metrics['precision']:.2%}")

    Raises:
        ValueError: If y_true and y_pred have different lengths, or if arrays are empty.

    Note:
        This function is completely model-agnostic! It works with predictions
        from ANY model architecture.
    """
    # Convert to numpy arrays for consistent handling
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length. "
                        f"Got {len(y_true)} and {len(y_pred)}")

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    # Handle y_prob if provided
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # If shape is (n_samples, 2), extract positive class probabilities
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  # Second column = positive class

    # Initialize results dictionary
    metrics = {}

    # 1. ACCURACY: Overall correctness
    #    (TP + TN) / (TP + TN + FP + FN)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # 2. PRECISION: Of predicted tumors, how many were correct?
    #    TP / (TP + FP)
    #    Low precision = many false alarms
    metrics['precision'] = precision_score(
        y_true, y_pred,
        pos_label=positive_label,
        zero_division=0  # Return 0 if no positive predictions
    )

    # 3. RECALL (SENSITIVITY): Of actual tumors, how many did we catch?
    #    TP / (TP + FN)
    #    THIS IS THE MOST CRITICAL METRIC!
    #    Low recall = missed tumors = dangerous!
    metrics['recall'] = recall_score(
        y_true, y_pred,
        pos_label=positive_label,
        zero_division=0  # Return 0 if no actual positives
    )

    # 4. F1 SCORE: Harmonic mean of precision and recall
    #    2 * (precision * recall) / (precision + recall)
    #    Balances both precision and recall
    metrics['f1'] = f1_score(
        y_true, y_pred,
        pos_label=positive_label,
        zero_division=0
    )

    # 5. CONFUSION MATRIX: Break down into TP, TN, FP, FN
    #    [[TN, FP],
    #     [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 6. SPECIFICITY (True Negative Rate): Of actual negatives, how many did we correctly identify?
    #    TN / (TN + FP)
    #    Complement of false positive rate
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    else:
        metrics['specificity'] = 0.0

    # 7. ROC-AUC: Area under the Receiver Operating Characteristic curve
    #    Measures model's ability to discriminate between classes
    #    Only computable if probabilities are provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            # Can fail if only one class is present in y_true
            print(f"[Metrics] Warning: Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    # 8. SUPPORT: Number of samples in each class
    #    Helps understand if classes are balanced
    metrics['support_positive'] = int(np.sum(y_true == positive_label))
    metrics['support_negative'] = int(np.sum(y_true != positive_label))

    # 9. Confusion matrix components (useful for debugging)
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)

    return metrics


def compute_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix components.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        tuple: (true_negatives, false_positives, false_negatives, true_positives)

    Example:
        >>> tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
        >>> print(f"True Positives (correctly identified tumors): {tp}")
        >>> print(f"False Negatives (missed tumors): {fn}")  # DANGEROUS!
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return int(tn), int(fp), int(fn), int(tp)


# =============================================================================
# METRIC INTERPRETATION AND REPORTING
# =============================================================================

def interpret_metrics(metrics: Dict[str, float], strict: bool = True) -> Dict[str, str]:
    """
    Provide clinical interpretation of metrics.

    This function translates raw metrics into actionable insights for the team,
    highlighting which metrics meet project goals and which need improvement.

    Args:
        metrics (dict): Metrics dictionary from compute_classification_metrics().
        strict (bool): If True, uses project's strict threshold of 0.80 for precision/recall.
            If False, uses more lenient thresholds.

    Returns:
        dict: Interpretation messages for each metric.

    Example:
        >>> metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        >>> interpretation = interpret_metrics(metrics)
        >>> for metric, message in interpretation.items():
        >>>     print(f"{metric}: {message}")
    """
    # Project goals from SMART objectives
    RECALL_TARGET = 0.80 if strict else 0.70
    PRECISION_TARGET = 0.80 if strict else 0.70
    F1_TARGET = 0.80 if strict else 0.70

    interpretation = {}

    # Recall interpretation (MOST CRITICAL)
    recall = metrics.get('recall', 0)
    if recall >= RECALL_TARGET:
        interpretation['recall'] = (
            f"✓ EXCELLENT - Recall of {recall:.2%} meets clinical safety threshold. "
            f"We're catching {recall:.2%} of tumors."
        )
    elif recall >= 0.70:
        interpretation['recall'] = (
            f"⚠ ACCEPTABLE BUT RISKY - Recall of {recall:.2%} is below target {RECALL_TARGET:.0%}. "
            f"We're missing {(1-recall):.2%} of tumors - this needs improvement!"
        )
    else:
        interpretation['recall'] = (
            f"✗ DANGEROUS - Recall of {recall:.2%} is critically low! "
            f"Missing {(1-recall):.2%} of tumors is clinically unacceptable."
        )

    # Precision interpretation
    precision = metrics.get('precision', 0)
    if precision >= PRECISION_TARGET:
        interpretation['precision'] = (
            f"✓ GOOD - Precision of {precision:.2%} meets target. "
            f"Low false positive rate."
        )
    elif precision >= 0.70:
        interpretation['precision'] = (
            f"⚠ NEEDS IMPROVEMENT - Precision of {precision:.2%} is below target {PRECISION_TARGET:.0%}. "
            f"Too many false positives."
        )
    else:
        interpretation['precision'] = (
            f"✗ POOR - Precision of {precision:.2%} means many false alarms. "
            f"Model is over-predicting tumors."
        )

    # F1 score interpretation
    f1 = metrics.get('f1', 0)
    if f1 >= F1_TARGET:
        interpretation['f1'] = f"✓ Strong F1 score of {f1:.2%} - good balance."
    else:
        interpretation['f1'] = f"⚠ F1 of {f1:.2%} below target - imbalanced precision/recall."

    # ROC-AUC interpretation
    roc_auc = metrics.get('roc_auc')
    if roc_auc is not None:
        if roc_auc >= 0.90:
            interpretation['roc_auc'] = f"✓ EXCELLENT discrimination ability (AUC={roc_auc:.3f})"
        elif roc_auc >= 0.80:
            interpretation['roc_auc'] = f"✓ Good discrimination ability (AUC={roc_auc:.3f})"
        elif roc_auc >= 0.70:
            interpretation['roc_auc'] = f"⚠ Fair discrimination ability (AUC={roc_auc:.3f})"
        else:
            interpretation['roc_auc'] = f"✗ Poor discrimination ability (AUC={roc_auc:.3f})"

    # Class balance check
    support_pos = metrics.get('support_positive', 0)
    support_neg = metrics.get('support_negative', 0)
    total = support_pos + support_neg
    if total > 0:
        imbalance_ratio = max(support_pos, support_neg) / min(support_pos, support_neg) if min(support_pos, support_neg) > 0 else float('inf')
        if imbalance_ratio > 3:
            interpretation['class_balance'] = (
                f"⚠ WARNING: Class imbalance detected! "
                f"Positive: {support_pos} ({support_pos/total:.1%}), "
                f"Negative: {support_neg} ({support_neg/total:.1%}). "
                f"Consider using class weights or resampling."
            )
        else:
            interpretation['class_balance'] = (
                f"✓ Classes reasonably balanced. "
                f"Positive: {support_pos} ({support_pos/total:.1%}), "
                f"Negative: {support_neg} ({support_neg/total:.1%})."
            )

    return interpretation


def print_metrics(
    metrics: Dict[str, float],
    show_interpretation: bool = True,
    show_confusion_matrix: bool = True
) -> None:
    """
    Pretty-print metrics in a formatted, readable way.

    Args:
        metrics (dict): Metrics from compute_classification_metrics().
        show_interpretation (bool): Whether to show clinical interpretation.
        show_confusion_matrix (bool): Whether to show confusion matrix breakdown.

    Example:
        >>> metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        >>> print_metrics(metrics)
        # Outputs nicely formatted table with all metrics
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS REPORT")
    print("=" * 70)

    # Core metrics
    print("\nCORE METRICS:")
    print(f"  Accuracy:   {metrics.get('accuracy', 0):>6.2%}  (Overall correctness)")
    print(f"  Precision:  {metrics.get('precision', 0):>6.2%}  (Positive predictive value)")
    print(f"  Recall:     {metrics.get('recall', 0):>6.2%}  ⚠ CRITICAL: Tumor detection rate!")
    print(f"  F1 Score:   {metrics.get('f1', 0):>6.2%}  (Harmonic mean of precision & recall)")
    print(f"  Specificity:{metrics.get('specificity', 0):>6.2%}  (True negative rate)")

    if metrics.get('roc_auc') is not None:
        print(f"  ROC-AUC:    {metrics.get('roc_auc', 0):>6.3f}  (Discrimination ability)")

    # Confusion matrix
    if show_confusion_matrix:
        print("\nCONFUSION MATRIX:")
        tn = metrics.get('true_negatives', 0)
        fp = metrics.get('false_positives', 0)
        fn = metrics.get('false_negatives', 0)
        tp = metrics.get('true_positives', 0)

        print(f"                 Predicted Negative  Predicted Positive")
        print(f"  Actual Negative       {tn:4d}                {fp:4d}")
        print(f"  Actual Positive       {fn:4d}                {tp:4d}")
        print(f"\n  True Positives:  {tp:4d}  (Correctly identified tumors)")
        print(f"  False Negatives: {fn:4d}  ⚠ MISSED TUMORS - most dangerous!")
        print(f"  False Positives: {fp:4d}  (Unnecessary scans/procedures)")
        print(f"  True Negatives:  {tn:4d}  (Correctly identified healthy)")

    # Class support
    print("\nCLASS DISTRIBUTION:")
    support_pos = metrics.get('support_positive', 0)
    support_neg = metrics.get('support_negative', 0)
    total = support_pos + support_neg
    if total > 0:
        print(f"  Positive cases (tumor):    {support_pos:4d} ({support_pos/total:>6.2%})")
        print(f"  Negative cases (no tumor): {support_neg:4d} ({support_neg/total:>6.2%})")

    # Interpretation
    if show_interpretation:
        print("\nCLINICAL INTERPRETATION:")
        interpretation = interpret_metrics(metrics)
        for key, message in interpretation.items():
            print(f"  {message}")

    # Project goal assessment
    print("\nPROJECT GOAL ASSESSMENT (SMART Objectives):")
    recall_goal = metrics.get('recall', 0) >= 0.80
    precision_goal = metrics.get('precision', 0) >= 0.80
    print(f"  Recall >= 80%:    {'✓ MET' if recall_goal else '✗ NOT MET'}")
    print(f"  Precision >= 80%: {'✓ MET' if precision_goal else '✗ NOT MET'}")

    if recall_goal and precision_goal:
        print("\n  🎉 CONGRATULATIONS! Model meets project goals!")
    else:
        print("\n  ⚠ Model needs improvement to meet project goals.")

    print("=" * 70 + "\n")


# =============================================================================
# SKLEARN COMPATIBILITY WRAPPERS
# =============================================================================

def get_classification_report(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    target_names: Optional[list] = None
) -> str:
    """
    Generate sklearn's classification report.

    This provides a different view of the same metrics, formatted as a table.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        target_names (list, optional): Names for classes. Default: ["No Tumor", "Tumor"].

    Returns:
        str: Formatted classification report.

    Example:
        >>> report = get_classification_report(y_true, y_pred)
        >>> print(report)
    """
    if target_names is None:
        target_names = ["No Tumor", "Tumor"]

    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=3
    )


# =============================================================================
# THRESHOLD OPTIMIZATION (ADVANCED)
# =============================================================================

def find_optimal_threshold(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    metric: str = "f1",
    min_recall: float = 0.80
) -> Tuple[float, float]:
    """
    Find optimal classification threshold to maximize a metric while maintaining minimum recall.

    By default, predictions use threshold=0.5 (predict tumor if probability > 0.5).
    But we can tune this! For clinical safety, we might want threshold=0.3 to catch more tumors.

    Args:
        y_true (array-like): Ground truth labels.
        y_prob (array-like): Prediction probabilities for positive class.
        metric (str): Metric to optimize: "f1", "precision", or "recall".
        min_recall (float): Minimum recall to maintain (default 0.80 for clinical safety).

    Returns:
        tuple: (optimal_threshold, metric_value)

    Example:
        >>> # Find threshold that maximizes F1 while keeping recall >= 0.80
        >>> optimal_thresh, f1_value = find_optimal_threshold(
        >>>     y_true, y_prob, metric="f1", min_recall=0.80
        >>> )
        >>> print(f"Use threshold {optimal_thresh:.2f} for F1={f1_value:.2%}")
        >>>
        >>> # Apply this threshold
        >>> y_pred_optimal = (y_prob >= optimal_thresh).astype(int)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Handle 2D probability arrays
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    # Try thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_metric_value = 0.0

    for thresh in thresholds:
        y_pred_temp = (y_prob >= thresh).astype(int)

        # Compute recall first to check constraint
        recall = recall_score(y_true, y_pred_temp, zero_division=0)

        # Skip if recall is below minimum
        if recall < min_recall:
            continue

        # Compute target metric
        if metric == "f1":
            metric_value = f1_score(y_true, y_pred_temp, zero_division=0)
        elif metric == "precision":
            metric_value = precision_score(y_true, y_pred_temp, zero_division=0)
        elif metric == "recall":
            metric_value = recall
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Update best if this is better
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = thresh

    return best_threshold, best_metric_value


# =============================================================================
# DOCUMENTATION
# =============================================================================
"""
USAGE EXAMPLES FOR TEAM:
=========================

Example 1: Basic metric computation
------------------------------------
from observability import metrics

# After model inference
y_true = test_labels  # Ground truth
y_pred = model.predict(test_images) > 0.5  # Binary predictions
y_prob = model.predict(test_images)  # Probabilities

# Compute all metrics
results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)

# Pretty print
metrics.print_metrics(results)


Example 2: Integration with MLflow
-----------------------------------
from observability import mlflow_config, metrics
import mlflow

mlflow_config.setup_mlflow()
mlflow_config.start_run(run_name="evaluate_baseline")

# Compute metrics
results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)

# Log to MLflow
mlflow.log_metrics({
    'test_accuracy': results['accuracy'],
    'test_precision': results['precision'],
    'test_recall': results['recall'],
    'test_f1': results['f1'],
    'test_roc_auc': results['roc_auc']
})

mlflow_config.end_run()


Example 3: Threshold optimization
----------------------------------
from observability import metrics

# Find best threshold
optimal_thresh, f1 = metrics.find_optimal_threshold(
    y_true, y_prob,
    metric="f1",
    min_recall=0.80  # MUST maintain 80% recall for safety
)

print(f"Optimal threshold: {optimal_thresh:.2f}")

# Use optimized threshold
y_pred_optimized = (y_prob >= optimal_thresh).astype(int)
results = metrics.compute_classification_metrics(y_true, y_pred_optimized, y_prob)
metrics.print_metrics(results)
"""
