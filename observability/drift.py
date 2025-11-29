"""
Drift Detection Module

ROLE IN OBSERVABILITY LAYER:
-----------------------------
This module detects when model behavior or input data changes over time - known as "drift."
Drift detection supports:
- MLO 8.7: Deployment monitoring and production model health
- MLO 14.1: Demonstrating awareness of production challenges
- Model lifecycle management: Knowing when to retrain

WHAT IS DRIFT?
--------------
Drift occurs when the statistical properties of input data or model predictions
change compared to training data or initial deployment. Types of drift:

1. PREDICTION DRIFT (Concept Drift):
   - Model outputs change over time
   - Example: Initially predicting 30% tumors, now predicting 60%
   - Causes: Different patient population, new scanner, image quality changes
   - Detection: Compare prediction distributions over time

2. EMBEDDING DRIFT (Input/Covariate Drift):
   - Input data characteristics change
   - Example: MRI images from new scanner look different
   - Detection: Compare feature representations (embeddings) from model
   - More sensitive than prediction drift

WHY DRIFT MATTERS CLINICALLY:
------------------------------
For brain tumor detection:
- Prediction drift might indicate:
  * Different patient population (more/fewer high-risk patients)
  * Scanner or protocol changes
  * Data quality degradation
  * Model degradation

- Undetected drift can lead to:
  * Decreased accuracy
  * More missed tumors (lower recall)
  * Unnecessary procedures (lower precision)
  * Need for model retraining

ARCHITECTURE-AGNOSTIC DESIGN:
------------------------------
This module works with:
- ANY model's predictions (just needs y_pred arrays)
- ANY model's embeddings/features (optional, if available)
- No assumptions about CNN architecture

How to connect embeddings:
- For basic CNNs: Use output of second-to-last layer
- For transfer learning: Use output of pre-trained backbone
- If unavailable: Prediction drift alone is still valuable!
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, chi2_contingency
import warnings


# =============================================================================
# PREDICTION DRIFT DETECTION
# =============================================================================

def detect_prediction_drift(
    reference_predictions: Union[np.ndarray, list],
    current_predictions: Union[np.ndarray, list],
    threshold: float = 0.1
) -> Dict[str, any]:
    """
    Detect drift in model prediction distributions.

    This compares the distribution of predictions between a reference period
    (e.g., training, initial deployment) and current period (e.g., this week's data).

    Method: Compare positive prediction rates and use statistical tests.

    Args:
        reference_predictions (array-like): Binary predictions from reference period.
            Values: 0 (no tumor) or 1 (tumor).
        current_predictions (array-like): Binary predictions from current period.
            Values: 0 or 1.
        threshold (float): Drift threshold for positive rate difference.
            Default: 0.1 (10% difference triggers drift warning).

    Returns:
        dict: Drift detection results containing:
            - 'drift_detected': Boolean indicating if drift exceeds threshold
            - 'reference_positive_rate': Fraction of positive predictions in reference
            - 'current_positive_rate': Fraction of positive predictions currently
            - 'positive_rate_diff': Absolute difference in positive rates
            - 'reference_size': Number of reference predictions
            - 'current_size': Number of current predictions
            - 'chi2_statistic': Chi-squared test statistic
            - 'chi2_pvalue': P-value for chi-squared test (low = significant drift)
            - 'severity': 'none', 'low', 'medium', or 'high'

    Example:
        >>> # Reference: predictions from training/validation
        >>> reference_preds = model.predict(train_data) > 0.5
        >>>
        >>> # Current: predictions from this week
        >>> current_preds = model.predict(production_data_this_week) > 0.5
        >>>
        >>> drift_result = detect_prediction_drift(reference_preds, current_preds)
        >>>
        >>> if drift_result['drift_detected']:
        >>>     print(f"DRIFT DETECTED! Severity: {drift_result['severity']}")
        >>>     print(f"Positive rate changed from {drift_result['reference_positive_rate']:.2%} "
        >>>           f"to {drift_result['current_positive_rate']:.2%}")

    Clinical Interpretation:
        - Positive rate increase: More tumors detected
          → Could indicate sicker patient population OR model false positives
        - Positive rate decrease: Fewer tumors detected
          → Could indicate healthier population OR model missing tumors
        - Large shift: Investigate immediately! Validate with radiologist review.
    """
    # Convert to numpy arrays
    ref_preds = np.asarray(reference_predictions)
    cur_preds = np.asarray(current_predictions)

    # Validate inputs
    if len(ref_preds) == 0 or len(cur_preds) == 0:
        raise ValueError("Both reference and current predictions must be non-empty")

    # Calculate positive prediction rates
    ref_positive_rate = np.mean(ref_preds)
    cur_positive_rate = np.mean(cur_preds)
    positive_rate_diff = abs(cur_positive_rate - ref_positive_rate)

    # Chi-squared test for distribution difference
    # Contingency table: [[ref_negative, ref_positive], [cur_negative, cur_positive]]
    ref_pos = int(np.sum(ref_preds))
    ref_neg = len(ref_preds) - ref_pos
    cur_pos = int(np.sum(cur_preds))
    cur_neg = len(cur_preds) - cur_pos

    contingency_table = np.array([[ref_neg, ref_pos], [cur_neg, cur_pos]])

    # Perform chi-squared test
    try:
        chi2, pvalue, _, _ = chi2_contingency(contingency_table)
    except ValueError as e:
        warnings.warn(f"Could not perform chi-squared test: {e}")
        chi2, pvalue = None, None

    # Determine drift severity
    if positive_rate_diff < threshold:
        severity = "none"
        drift_detected = False
    elif positive_rate_diff < threshold * 2:
        severity = "low"
        drift_detected = True
    elif positive_rate_diff < threshold * 3:
        severity = "medium"
        drift_detected = True
    else:
        severity = "high"
        drift_detected = True

    # Also check statistical significance (p-value < 0.05)
    if pvalue is not None and pvalue < 0.05:
        if severity == "none":
            severity = "low"
            drift_detected = True

    return {
        'drift_detected': drift_detected,
        'reference_positive_rate': float(ref_positive_rate),
        'current_positive_rate': float(cur_positive_rate),
        'positive_rate_diff': float(positive_rate_diff),
        'reference_size': len(ref_preds),
        'current_size': len(cur_preds),
        'chi2_statistic': float(chi2) if chi2 is not None else None,
        'chi2_pvalue': float(pvalue) if pvalue is not None else None,
        'severity': severity,
    }


def detect_continuous_prediction_drift(
    reference_probabilities: Union[np.ndarray, list],
    current_probabilities: Union[np.ndarray, list],
    threshold_ks: float = 0.1
) -> Dict[str, any]:
    """
    Detect drift in continuous prediction probabilities using KS test.

    While detect_prediction_drift() works with binary predictions, this function
    works with continuous probability outputs (e.g., [0.12, 0.89, 0.34, ...]).
    This is more sensitive to subtle distribution changes.

    Method: Kolmogorov-Smirnov (KS) test to compare probability distributions.

    Args:
        reference_probabilities (array-like): Prediction probabilities from reference period.
            Values in [0, 1], typically probability of positive class (tumor).
        current_probabilities (array-like): Prediction probabilities from current period.
        threshold_ks (float): KS statistic threshold. Default: 0.1.
            KS statistic ranges from 0 (identical) to 1 (completely different).

    Returns:
        dict: Drift detection results containing:
            - 'drift_detected': Boolean
            - 'ks_statistic': KS test statistic (0-1 scale)
            - 'ks_pvalue': P-value (low = significant difference)
            - 'reference_mean_prob': Mean probability in reference
            - 'current_mean_prob': Mean probability currently
            - 'severity': 'none', 'low', 'medium', or 'high'

    Example:
        >>> # Get continuous probabilities from model
        >>> reference_probs = model.predict(train_data)  # Shape: (n, 1) or (n, 2)
        >>> current_probs = model.predict(production_data)
        >>>
        >>> # If shape is (n, 2), extract positive class probabilities
        >>> if reference_probs.shape[1] == 2:
        >>>     reference_probs = reference_probs[:, 1]
        >>>     current_probs = current_probs[:, 1]
        >>>
        >>> drift_result = detect_continuous_prediction_drift(reference_probs, current_probs)

    Note:
        This is more sensitive than binary prediction drift, but requires
        probability outputs from your model (not just binary predictions).
    """
    # Convert to numpy arrays
    ref_probs = np.asarray(reference_probabilities)
    cur_probs = np.asarray(current_probabilities)

    # Validate
    if len(ref_probs) == 0 or len(cur_probs) == 0:
        raise ValueError("Both reference and current probabilities must be non-empty")

    # KS test: compares cumulative distributions
    ks_statistic, ks_pvalue = ks_2samp(ref_probs, cur_probs)

    # Mean probabilities
    ref_mean_prob = float(np.mean(ref_probs))
    cur_mean_prob = float(np.mean(cur_probs))

    # Determine severity based on KS statistic
    if ks_statistic < threshold_ks:
        severity = "none"
        drift_detected = False
    elif ks_statistic < threshold_ks * 2:
        severity = "low"
        drift_detected = True
    elif ks_statistic < threshold_ks * 3:
        severity = "medium"
        drift_detected = True
    else:
        severity = "high"
        drift_detected = True

    # Also check statistical significance
    if ks_pvalue < 0.05 and severity == "none":
        severity = "low"
        drift_detected = True

    return {
        'drift_detected': drift_detected,
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'reference_mean_prob': ref_mean_prob,
        'current_mean_prob': cur_mean_prob,
        'severity': severity,
    }


# =============================================================================
# EMBEDDING DRIFT DETECTION
# =============================================================================

def detect_embedding_drift(
    reference_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    method: str = "cosine",
    threshold: float = 0.3
) -> Dict[str, any]:
    """
    Detect drift in model embeddings/feature representations.

    Embeddings are internal representations learned by the model. Comparing
    embedding distributions is more sensitive than comparing predictions,
    as it can detect input data changes even before predictions change.

    IMPORTANT: This requires extracting embeddings from your model.
    HOW TO GET EMBEDDINGS (Architecture-Agnostic):
    -----------------------------------------------
    1. Basic CNN: Use output of second-to-last layer (before final dense layer)
    2. Transfer learning (ResNet, EfficientNet, etc.): Use output of backbone
    3. Custom model: Any intermediate layer that captures image features

    Example extraction:
        # Keras/TensorFlow
        embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        embeddings = embedding_model.predict(images)

        # PyTorch
        def get_embeddings(model, images):
            with torch.no_grad():
                features = model.feature_extractor(images)  # Before classifier
            return features.numpy()

    Args:
        reference_embeddings (np.ndarray): Embeddings from reference period.
            Shape: (n_samples, embedding_dim). Example: (1000, 512).
        current_embeddings (np.ndarray): Embeddings from current period.
            Shape: (m_samples, embedding_dim). Must have same embedding_dim.
        method (str): Drift detection method:
            - "cosine": Compare mean embedding vectors using cosine distance
            - "mmd": Maximum Mean Discrepancy (more sophisticated)
        threshold (float): Drift threshold. For cosine: 0.3 is reasonable default.

    Returns:
        dict: Drift detection results:
            - 'drift_detected': Boolean
            - 'drift_score': Numeric drift score (interpretation depends on method)
            - 'method': Method used
            - 'threshold': Threshold used
            - 'severity': 'none', 'low', 'medium', or 'high'

    Example:
        >>> # Extract embeddings from model
        >>> embedding_model = create_embedding_extractor(model)
        >>> ref_embeddings = embedding_model.predict(train_images)
        >>> cur_embeddings = embedding_model.predict(production_images)
        >>>
        >>> # Detect drift
        >>> drift_result = detect_embedding_drift(ref_embeddings, cur_embeddings)
        >>>
        >>> if drift_result['drift_detected']:
        >>>     print(f"Embedding drift detected! Score: {drift_result['drift_score']:.3f}")

    Note:
        If you don't have embeddings yet, that's OK! Focus on prediction drift first.
        Add embedding drift later as your model matures.
    """
    # Validate inputs
    if reference_embeddings.ndim != 2 or current_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays (n_samples, embedding_dim)")

    if reference_embeddings.shape[1] != current_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions must match. Got {reference_embeddings.shape[1]} "
            f"and {current_embeddings.shape[1]}"
        )

    # Calculate drift based on method
    if method == "cosine":
        # Compare mean embedding vectors
        ref_mean = np.mean(reference_embeddings, axis=0)
        cur_mean = np.mean(current_embeddings, axis=0)

        # Cosine distance (0 = identical, 2 = opposite)
        # We use 1 - cosine_similarity for drift score
        drift_score = float(cosine(ref_mean, cur_mean))

    elif method == "mmd":
        # Maximum Mean Discrepancy (more sophisticated but slower)
        drift_score = _compute_mmd(reference_embeddings, current_embeddings)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'cosine' or 'mmd'.")

    # Determine severity
    if drift_score < threshold:
        severity = "none"
        drift_detected = False
    elif drift_score < threshold * 1.5:
        severity = "low"
        drift_detected = True
    elif drift_score < threshold * 2:
        severity = "medium"
        drift_detected = True
    else:
        severity = "high"
        drift_detected = True

    return {
        'drift_detected': drift_detected,
        'drift_score': drift_score,
        'method': method,
        'threshold': threshold,
        'severity': severity,
    }


def _compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None
) -> float:
    """
    Compute Maximum Mean Discrepancy between two sets of embeddings.

    MMD is a kernel-based method for comparing distributions. It's more
    sophisticated than simple mean comparison but also more computationally expensive.

    Args:
        X (np.ndarray): First set of embeddings, shape (n_samples, dim).
        Y (np.ndarray): Second set of embeddings, shape (m_samples, dim).
        kernel (str): Kernel type. Currently only "rbf" supported.
        gamma (float, optional): RBF kernel parameter. If None, uses 1/dim.

    Returns:
        float: MMD statistic (higher = more different).
    """
    n = X.shape[0]
    m = Y.shape[0]

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # RBF kernel: k(x, y) = exp(-gamma * ||x - y||^2)
    def rbf_kernel(A, B):
        # Compute pairwise squared distances
        dist_sq = np.sum(A**2, axis=1, keepdims=True) + \
                  np.sum(B**2, axis=1) - 2 * np.dot(A, B.T)
        return np.exp(-gamma * dist_sq)

    # Compute kernel matrices
    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    # MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    mmd_sq = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) + \
             (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1)) - \
             2 * np.mean(K_XY)

    # Return square root (actual MMD, not squared)
    return float(np.sqrt(max(mmd_sq, 0)))  # max() handles numerical issues


# =============================================================================
# COMBINED DRIFT MONITORING
# =============================================================================

def monitor_drift(
    reference_predictions: Union[np.ndarray, list],
    current_predictions: Union[np.ndarray, list],
    reference_embeddings: Optional[np.ndarray] = None,
    current_embeddings: Optional[np.ndarray] = None,
    prediction_threshold: float = 0.1,
    embedding_threshold: float = 0.3
) -> Dict[str, any]:
    """
    Comprehensive drift monitoring combining prediction and embedding drift.

    This is the main function to use for production drift monitoring.
    It combines multiple drift detection methods and provides overall assessment.

    Args:
        reference_predictions (array-like): Binary predictions from reference period.
        current_predictions (array-like): Binary predictions from current period.
        reference_embeddings (np.ndarray, optional): Reference embeddings.
        current_embeddings (np.ndarray, optional): Current embeddings.
        prediction_threshold (float): Threshold for prediction drift (default: 0.1).
        embedding_threshold (float): Threshold for embedding drift (default: 0.3).

    Returns:
        dict: Comprehensive drift report:
            - 'overall_drift_detected': Boolean (ANY drift detected)
            - 'prediction_drift': Results from prediction drift detection
            - 'embedding_drift': Results from embedding drift detection (if provided)
            - 'severity': Overall severity (max of individual severities)
            - 'recommendations': List of recommended actions

    Example:
        >>> # Monitor drift weekly
        >>> weekly_report = monitor_drift(
        >>>     reference_predictions=baseline_preds,
        >>>     current_predictions=this_week_preds,
        >>>     reference_embeddings=baseline_embeddings,
        >>>     current_embeddings=this_week_embeddings
        >>> )
        >>>
        >>> if weekly_report['overall_drift_detected']:
        >>>     print(f"Drift severity: {weekly_report['severity']}")
        >>>     for rec in weekly_report['recommendations']:
        >>>         print(f"- {rec}")
    """
    report = {}

    # Prediction drift (always performed)
    pred_drift = detect_prediction_drift(
        reference_predictions,
        current_predictions,
        threshold=prediction_threshold
    )
    report['prediction_drift'] = pred_drift

    # Embedding drift (if embeddings provided)
    if reference_embeddings is not None and current_embeddings is not None:
        emb_drift = detect_embedding_drift(
            reference_embeddings,
            current_embeddings,
            threshold=embedding_threshold
        )
        report['embedding_drift'] = emb_drift
    else:
        report['embedding_drift'] = None

    # Overall assessment
    severities = [pred_drift['severity']]
    if report['embedding_drift']:
        severities.append(report['embedding_drift']['severity'])

    # Determine overall severity (take maximum)
    severity_order = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    max_severity = max(severities, key=lambda s: severity_order[s])
    report['severity'] = max_severity
    report['overall_drift_detected'] = max_severity != 'none'

    # Generate recommendations
    recommendations = []
    if max_severity == 'none':
        recommendations.append("No significant drift detected. Continue monitoring.")
    elif max_severity == 'low':
        recommendations.append("Low drift detected. Increase monitoring frequency.")
        recommendations.append("Review recent prediction samples.")
    elif max_severity == 'medium':
        recommendations.append("Moderate drift detected. Investigate cause:")
        recommendations.append("  - Check for changes in data source or scanner")
        recommendations.append("  - Review data quality metrics")
        recommendations.append("  - Consider model evaluation on recent data")
    else:  # high
        recommendations.append("⚠ HIGH DRIFT DETECTED! IMMEDIATE ACTION REQUIRED:")
        recommendations.append("  - Halt automated predictions if safety-critical")
        recommendations.append("  - Investigate data pipeline changes")
        recommendations.append("  - Evaluate model performance on recent data")
        recommendations.append("  - Consider model retraining")
        recommendations.append("  - Consult with domain experts (radiologists)")

    report['recommendations'] = recommendations

    return report


def print_drift_report(report: Dict[str, any]) -> None:
    """
    Pretty-print a drift monitoring report.

    Args:
        report (dict): Report from monitor_drift().

    Example:
        >>> report = monitor_drift(ref_preds, cur_preds)
        >>> print_drift_report(report)
    """
    print("\n" + "=" * 70)
    print("DRIFT MONITORING REPORT")
    print("=" * 70)

    # Overall status
    print(f"\nOVERALL STATUS:")
    if report['overall_drift_detected']:
        print(f"  ⚠ DRIFT DETECTED - Severity: {report['severity'].upper()}")
    else:
        print(f"  ✓ No significant drift detected")

    # Prediction drift details
    print(f"\nPREDICTION DRIFT:")
    pred = report['prediction_drift']
    print(f"  Reference positive rate: {pred['reference_positive_rate']:>6.2%}")
    print(f"  Current positive rate:   {pred['current_positive_rate']:>6.2%}")
    print(f"  Difference:              {pred['positive_rate_diff']:>6.2%}")
    if pred['chi2_pvalue'] is not None:
        print(f"  Chi-squared p-value:     {pred['chi2_pvalue']:>6.4f}")
    print(f"  Severity:                {pred['severity']}")

    # Embedding drift details (if available)
    if report['embedding_drift']:
        print(f"\nEMBEDDING DRIFT:")
        emb = report['embedding_drift']
        print(f"  Method:       {emb['method']}")
        print(f"  Drift score:  {emb['drift_score']:.4f}")
        print(f"  Threshold:    {emb['threshold']:.4f}")
        print(f"  Severity:     {emb['severity']}")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  {rec}")

    print("=" * 70 + "\n")


# =============================================================================
# DOCUMENTATION
# =============================================================================
"""
USAGE EXAMPLES FOR TEAM:
=========================

Example 1: Basic prediction drift monitoring
---------------------------------------------
from observability import drift

# Get predictions from two periods
baseline_preds = model.predict(train_data) > 0.5
current_preds = model.predict(production_data) > 0.5

# Check for drift
drift_result = drift.detect_prediction_drift(baseline_preds, current_preds)

if drift_result['drift_detected']:
    print(f"Drift severity: {drift_result['severity']}")


Example 2: Continuous probability drift
----------------------------------------
from observability import drift

# Use continuous probabilities for more sensitive detection
baseline_probs = model.predict(train_data)[:, 1]  # Positive class prob
current_probs = model.predict(production_data)[:, 1]

drift_result = drift.detect_continuous_prediction_drift(baseline_probs, current_probs)


Example 3: Embedding drift (advanced)
--------------------------------------
from observability import drift

# Extract embeddings from your model
# (This part is model-specific but the drift detection is not!)
embedding_model = create_embedding_extractor(your_model)

baseline_emb = embedding_model.predict(train_data)
current_emb = embedding_model.predict(production_data)

drift_result = drift.detect_embedding_drift(baseline_emb, current_emb)


Example 4: Comprehensive monitoring (recommended)
--------------------------------------------------
from observability import drift

# Monitor both prediction and embedding drift
report = drift.monitor_drift(
    reference_predictions=baseline_preds,
    current_predictions=this_week_preds,
    reference_embeddings=baseline_embeddings,  # Optional
    current_embeddings=this_week_embeddings     # Optional
)

drift.print_drift_report(report)

# Take action based on severity
if report['severity'] in ['medium', 'high']:
    send_alert_to_team(report)
    trigger_model_evaluation()


Example 5: Weekly drift monitoring pipeline
--------------------------------------------
from observability import drift, mlflow_config
import mlflow

mlflow_config.setup_mlflow()
mlflow_config.start_run(run_name=f"drift_check_week_{week_num}")

# Load baseline (computed once from training data)
baseline = load_baseline_data()

# Get this week's predictions
week_predictions = get_weekly_predictions()

# Monitor drift
report = drift.monitor_drift(
    reference_predictions=baseline['predictions'],
    current_predictions=week_predictions
)

# Log to MLflow
mlflow.log_metrics({
    'drift_positive_rate_diff': report['prediction_drift']['positive_rate_diff'],
    'drift_severity_numeric': {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[report['severity']]
})

mlflow_config.end_run()
"""
