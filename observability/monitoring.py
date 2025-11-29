"""
End-to-End Monitoring Module

ROLE IN OBSERVABILITY LAYER:
-----------------------------
This module provides high-level monitoring utilities that tie together all
other observability components. It supports:
- MLO 14.1: Complete end-to-end demonstration of observability pipeline
- MLO 8.7: Production-grade monitoring implementation
- Integration of metrics, drift, quality, and alerts

PURPOSE:
--------
While other modules provide specific functionality (metrics, drift, etc.),
this module provides:
1. End-to-end monitoring workflows
2. Batch evaluation utilities
3. Simulation utilities for demonstration
4. Example monitoring pipelines

This is especially useful for:
- Course demonstrations and presentations
- Testing the observability stack
- Providing templates for production deployment

ARCHITECTURE-AGNOSTIC:
----------------------
All functions work with generic inputs (predictions, labels, latencies)
without assumptions about model architecture.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
from pathlib import Path


# =============================================================================
# BATCH EVALUATION UTILITIES
# =============================================================================

def evaluate_batch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    batch_name: str = "batch",
    reference_predictions: Optional[np.ndarray] = None,
    check_alerts: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a batch of predictions.

    This function performs a complete evaluation pipeline:
    1. Compute classification metrics
    2. Check for drift (if reference provided)
    3. Generate alerts (if enabled)
    4. Provide summary report

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).
        y_prob (np.ndarray, optional): Prediction probabilities.
        batch_name (str): Descriptive name for this batch.
        reference_predictions (np.ndarray, optional): Reference predictions for drift detection.
        check_alerts (bool): Whether to check for alerts (default: True).

    Returns:
        dict: Comprehensive evaluation report containing:
            - 'batch_name': Name of batch
            - 'metrics': Performance metrics
            - 'drift': Drift detection results (if reference provided)
            - 'alerts': List of alerts (if check_alerts=True)
            - 'summary': Human-readable summary

    Example:
        >>> from observability import monitoring
        >>>
        >>> # Evaluate test set
        >>> report = monitoring.evaluate_batch(
        >>>     y_true=test_labels,
        >>>     y_pred=test_predictions,
        >>>     y_prob=test_probabilities,
        >>>     batch_name="test_set_evaluation",
        >>>     reference_predictions=train_predictions  # For drift check
        >>> )
        >>>
        >>> # Print summary
        >>> print(report['summary'])
        >>>
        >>> # Access specific components
        >>> if report['alerts']:
        >>>     print(f"Found {len(report['alerts'])} alerts!")
    """
    from observability import metrics as metrics_module
    from observability import drift as drift_module
    from observability import alerts as alerts_module

    report = {
        'batch_name': batch_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': len(y_true)
    }

    # 1. Compute classification metrics
    print(f"[Monitoring] Evaluating batch '{batch_name}' ({len(y_true)} samples)...")
    metrics = metrics_module.compute_classification_metrics(y_true, y_pred, y_prob)
    report['metrics'] = metrics

    # 2. Check for drift (if reference provided)
    if reference_predictions is not None:
        print(f"[Monitoring] Checking for drift...")
        drift_report = drift_module.monitor_drift(
            reference_predictions=reference_predictions,
            current_predictions=y_pred
        )
        report['drift'] = drift_report
    else:
        report['drift'] = None

    # 3. Check for alerts
    if check_alerts:
        print(f"[Monitoring] Checking alerts...")
        alert_list = alerts_module.check_all_alerts(
            metrics=metrics,
            drift_report=report.get('drift')
        )
        report['alerts'] = alert_list
    else:
        report['alerts'] = []

    # 4. Generate summary
    summary_lines = []
    summary_lines.append(f"Batch Evaluation: {batch_name}")
    summary_lines.append(f"Samples: {len(y_true)}")
    summary_lines.append(f"\nPerformance:")
    summary_lines.append(f"  Accuracy:  {metrics['accuracy']:.2%}")
    summary_lines.append(f"  Precision: {metrics['precision']:.2%}")
    summary_lines.append(f"  Recall:    {metrics['recall']:.2%}")
    summary_lines.append(f"  F1 Score:  {metrics['f1']:.2%}")

    if report['drift']:
        drift = report['drift']['prediction_drift']
        summary_lines.append(f"\nDrift:")
        summary_lines.append(f"  Positive rate: {drift['reference_positive_rate']:.2%} → "
                           f"{drift['current_positive_rate']:.2%}")
        summary_lines.append(f"  Status: {'DRIFT DETECTED' if drift['drift_detected'] else 'No drift'}")

    if report['alerts']:
        summary_lines.append(f"\nAlerts: {len(report['alerts'])} alert(s)")
        critical = sum(1 for a in report['alerts'] if a.severity == 'critical')
        if critical > 0:
            summary_lines.append(f"  ⚠️ {critical} CRITICAL alert(s)!")
    else:
        summary_lines.append(f"\nAlerts: None - all systems normal")

    report['summary'] = '\n'.join(summary_lines)

    print(f"[Monitoring] Evaluation complete.")
    return report


def evaluate_multiple_batches(
    batches: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    reference_predictions: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate multiple batches and compare results.

    Useful for comparing performance across different datasets (train, val, test)
    or different time periods.

    Args:
        batches (list): List of (name, y_true, y_pred, y_prob) tuples.
        reference_predictions (np.ndarray, optional): Reference for drift detection.

    Returns:
        dict: Reports for all batches plus comparison summary.

    Example:
        >>> batches = [
        >>>     ("train", train_labels, train_preds, train_probs),
        >>>     ("val", val_labels, val_preds, val_probs),
        >>>     ("test", test_labels, test_preds, test_probs)
        >>> ]
        >>>
        >>> results = monitoring.evaluate_multiple_batches(batches)
        >>>
        >>> # Compare performance
        >>> for batch_name, report in results['batch_reports'].items():
        >>>     print(f"{batch_name}: Recall={report['metrics']['recall']:.2%}")
    """
    batch_reports = {}

    for batch_name, y_true, y_pred, y_prob in batches:
        report = evaluate_batch(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            batch_name=batch_name,
            reference_predictions=reference_predictions
        )
        batch_reports[batch_name] = report

    # Comparison summary
    comparison = {
        'batch_reports': batch_reports,
        'num_batches': len(batches),
        'metric_comparison': {}
    }

    # Compare key metrics across batches
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        comparison['metric_comparison'][metric_name] = {
            batch_name: report['metrics'][metric_name]
            for batch_name, report in batch_reports.items()
        }

    return comparison


# =============================================================================
# LATENCY MONITORING
# =============================================================================

def simulate_latency_samples(
    num_samples: int = 1000,
    mean_ms: float = 100,
    std_ms: float = 20,
    add_outliers: bool = True,
    outlier_probability: float = 0.05
) -> np.ndarray:
    """
    Simulate inference latency samples for demonstration.

    This generates realistic latency data with optional outliers,
    useful for demonstrating monitoring capabilities in presentations.

    Args:
        num_samples (int): Number of latency samples to generate.
        mean_ms (float): Mean latency in milliseconds.
        std_ms (float): Standard deviation of latency.
        add_outliers (bool): Whether to add occasional slow outliers.
        outlier_probability (float): Probability of outlier (if add_outliers=True).

    Returns:
        np.ndarray: Array of latency values in milliseconds.

    Example:
        >>> from observability import monitoring
        >>>
        >>> # Simulate healthy latency
        >>> latencies = monitoring.simulate_latency_samples(
        >>>     num_samples=1000, mean_ms=100, std_ms=20
        >>> )
        >>>
        >>> # Analyze
        >>> stats = monitoring.compute_latency_stats(latencies)
        >>> print(f"p95 latency: {stats['p95']:.1f}ms")
    """
    # Generate normal latency distribution
    latencies = np.random.normal(mean_ms, std_ms, num_samples)

    # Add outliers (occasional very slow requests)
    if add_outliers:
        outlier_mask = np.random.random(num_samples) < outlier_probability
        num_outliers = np.sum(outlier_mask)
        if num_outliers > 0:
            # Outliers are 3-10x slower than mean
            outlier_multipliers = np.random.uniform(3, 10, num_outliers)
            latencies[outlier_mask] = mean_ms * outlier_multipliers

    # Ensure no negative latencies
    latencies = np.maximum(latencies, 1)

    return latencies


def compute_latency_stats(latencies: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive latency statistics.

    Args:
        latencies (np.ndarray): Array of latency values in milliseconds.

    Returns:
        dict: Statistics including mean, median, p50, p95, p99, min, max.

    Example:
        >>> latencies = [100, 120, 95, 110, 105, 450, 98, 102]  # One outlier
        >>> stats = compute_latency_stats(np.array(latencies))
        >>> print(f"Mean: {stats['mean']:.1f}ms, p95: {stats['p95']:.1f}ms")
    """
    return {
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p90': float(np.percentile(latencies, 90)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
        'num_samples': len(latencies)
    }


# =============================================================================
# DATA GENERATION FOR DEMONSTRATION
# =============================================================================

def generate_simulated_predictions(
    num_samples: int = 1000,
    true_positive_rate: float = 0.30,
    model_recall: float = 0.85,
    model_precision: float = 0.82
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate simulated ground truth and predictions for demonstration.

    This creates realistic classification data for testing and demonstrating
    the observability stack without needing a trained model.

    Args:
        num_samples (int): Number of samples to generate.
        true_positive_rate (float): Fraction of samples that are actually positive (tumor).
        model_recall (float): Desired recall (sensitivity) of simulated model.
        model_precision (float): Desired precision of simulated model.

    Returns:
        tuple: (y_true, y_pred, y_prob)
            - y_true: Ground truth labels (0 or 1)
            - y_pred: Predicted labels (0 or 1)
            - y_prob: Prediction probabilities for positive class

    Example:
        >>> from observability import monitoring, metrics
        >>>
        >>> # Generate realistic data
        >>> y_true, y_pred, y_prob = monitoring.generate_simulated_predictions(
        >>>     num_samples=1000,
        >>>     true_positive_rate=0.30,
        >>>     model_recall=0.85,
        >>>     model_precision=0.82
        >>> )
        >>>
        >>> # Evaluate
        >>> results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)
        >>> print(f"Recall: {results['recall']:.2%}")  # Should be ~85%
    """
    # Generate ground truth
    num_positive = int(num_samples * true_positive_rate)
    y_true = np.zeros(num_samples, dtype=int)
    y_true[:num_positive] = 1
    np.random.shuffle(y_true)

    # Generate predictions to achieve target recall and precision
    # Recall = TP / (TP + FN), so TP = recall * num_positives
    # Precision = TP / (TP + FP), so FP = TP/precision - TP

    num_true_positives = int(num_positive * model_recall)
    num_false_negatives = num_positive - num_true_positives

    # From precision: TP / (TP + FP) = precision
    # So: TP + FP = TP / precision
    # Therefore: FP = TP / precision - TP
    num_predicted_positive = int(num_true_positives / model_precision)
    num_false_positives = num_predicted_positive - num_true_positives

    # Build predictions
    y_pred = np.zeros(num_samples, dtype=int)

    # Get indices of positive and negative samples
    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    # Randomly select which positives to correctly identify (TP)
    tp_indices = np.random.choice(positive_indices, num_true_positives, replace=False)
    y_pred[tp_indices] = 1

    # Randomly select which negatives to incorrectly label as positive (FP)
    if num_false_positives > 0 and len(negative_indices) >= num_false_positives:
        fp_indices = np.random.choice(negative_indices, num_false_positives, replace=False)
        y_pred[fp_indices] = 1

    # Generate probabilities consistent with predictions
    # Positive predictions should have high probability, negative predictions low
    y_prob = np.zeros(num_samples)

    # For predicted positives: sample from high probability distribution
    pred_positive_mask = y_pred == 1
    y_prob[pred_positive_mask] = np.random.beta(8, 2, size=np.sum(pred_positive_mask))

    # For predicted negatives: sample from low probability distribution
    pred_negative_mask = y_pred == 0
    y_prob[pred_negative_mask] = np.random.beta(2, 8, size=np.sum(pred_negative_mask))

    return y_true, y_pred, y_prob


# =============================================================================
# COMPLETE MONITORING DEMONSTRATION
# =============================================================================

def run_monitoring_demo(verbose: bool = True) -> Dict[str, Any]:
    """
    Run a complete observability monitoring demonstration.

    This function demonstrates all observability capabilities:
    1. Simulated model predictions
    2. Metric computation
    3. Drift detection
    4. Data quality checks (simulated)
    5. Latency monitoring
    6. Alert generation
    7. Comprehensive reporting

    Perfect for course presentations and demonstrations.

    Args:
        verbose (bool): Whether to print detailed output.

    Returns:
        dict: Complete monitoring results including all reports and visualizations.

    Example:
        >>> from observability import monitoring
        >>>
        >>> # Run complete demo
        >>> demo_results = monitoring.run_monitoring_demo(verbose=True)
        >>>
        >>> # Access components
        >>> print(demo_results['baseline_report']['summary'])
        >>> print(demo_results['production_report']['summary'])
    """
    from observability import metrics as metrics_module
    from observability import drift as drift_module
    from observability import alerts as alerts_module

    if verbose:
        print("\n" + "=" * 70)
        print("OBSERVABILITY MONITORING DEMONSTRATION")
        print("Brain MRI Tumor Classification - Johns Hopkins 635.603")
        print("=" * 70 + "\n")

    results = {}

    # =========================================================================
    # STEP 1: Generate baseline data (simulating training/validation data)
    # =========================================================================
    if verbose:
        print("STEP 1: Generating baseline predictions...")

    baseline_y_true, baseline_y_pred, baseline_y_prob = generate_simulated_predictions(
        num_samples=1000,
        true_positive_rate=0.30,
        model_recall=0.88,
        model_precision=0.85
    )

    baseline_report = evaluate_batch(
        y_true=baseline_y_true,
        y_pred=baseline_y_pred,
        y_prob=baseline_y_prob,
        batch_name="baseline_validation_set",
        check_alerts=True
    )

    results['baseline_report'] = baseline_report

    if verbose:
        print("\nBaseline Performance:")
        print(f"  Recall:    {baseline_report['metrics']['recall']:.2%}")
        print(f"  Precision: {baseline_report['metrics']['precision']:.2%}")
        print(f"  F1 Score:  {baseline_report['metrics']['f1']:.2%}")

    # =========================================================================
    # STEP 2: Simulate production data with slight drift
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 2: Generating production predictions (with slight drift)...")

    # Simulate production data with slightly degraded performance
    production_y_true, production_y_pred, production_y_prob = generate_simulated_predictions(
        num_samples=500,
        true_positive_rate=0.35,  # Slightly different patient population
        model_recall=0.78,         # Degraded recall - ALERT!
        model_precision=0.83       # Slightly degraded precision
    )

    production_report = evaluate_batch(
        y_true=production_y_true,
        y_pred=production_y_pred,
        y_prob=production_y_prob,
        batch_name="production_week_1",
        reference_predictions=baseline_y_pred,  # Compare to baseline
        check_alerts=True
    )

    results['production_report'] = production_report

    if verbose:
        print("\nProduction Performance:")
        print(f"  Recall:    {production_report['metrics']['recall']:.2%}")
        print(f"  Precision: {production_report['metrics']['precision']:.2%}")
        print(f"  F1 Score:  {production_report['metrics']['f1']:.2%}")

        if production_report['drift']:
            drift = production_report['drift']['prediction_drift']
            print(f"\nDrift Status:")
            print(f"  {'DRIFT DETECTED' if drift['drift_detected'] else 'No drift'}")
            print(f"  Positive rate: {drift['reference_positive_rate']:.2%} → "
                  f"{drift['current_positive_rate']:.2%}")

    # =========================================================================
    # STEP 3: Latency monitoring
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 3: Simulating inference latency monitoring...")

    latencies = simulate_latency_samples(
        num_samples=1000,
        mean_ms=150,
        std_ms=30,
        add_outliers=True
    )

    latency_stats = compute_latency_stats(latencies)
    results['latency_stats'] = latency_stats

    if verbose:
        print(f"\nLatency Statistics:")
        print(f"  Mean:   {latency_stats['mean']:.1f} ms")
        print(f"  Median: {latency_stats['median']:.1f} ms")
        print(f"  p95:    {latency_stats['p95']:.1f} ms")
        print(f"  p99:    {latency_stats['p99']:.1f} ms")
        print(f"  Max:    {latency_stats['max']:.1f} ms")

    # Check for latency alerts
    latency_alerts = alerts_module.check_latency_alerts(latency_stats)
    results['latency_alerts'] = latency_alerts

    # =========================================================================
    # STEP 4: Comprehensive alert summary
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 4: Alert Summary...")

    all_alerts = production_report['alerts'] + latency_alerts
    results['all_alerts'] = all_alerts

    if verbose:
        alerts_module.print_alerts(all_alerts)

    # =========================================================================
    # STEP 5: Generate recommendations
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("RECOMMENDATIONS:")

    recommendations = []

    # Check recall
    if production_report['metrics']['recall'] < 0.80:
        recommendations.append(
            "⚠️  CRITICAL: Recall below 80% threshold - model is missing tumors! "
            "Recommend immediate investigation and possible model retraining."
        )

    # Check drift
    if production_report['drift'] and production_report['drift']['prediction_drift']['drift_detected']:
        recommendations.append(
            "⚠️  Prediction drift detected. Review recent data for quality issues "
            "or patient population changes."
        )

    # Check for improvement potential
    if baseline_report['metrics']['recall'] > production_report['metrics']['recall']:
        diff = baseline_report['metrics']['recall'] - production_report['metrics']['recall']
        recommendations.append(
            f"📉 Recall degraded by {diff:.2%} since baseline. "
            f"Model may need retraining or threshold adjustment."
        )

    if not recommendations:
        recommendations.append("✅ All systems operating normally. Continue monitoring.")

    results['recommendations'] = recommendations

    if verbose:
        for rec in recommendations:
            print(f"  {rec}")

    # =========================================================================
    # STEP 6: Final summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated comprehensive observability report covering:")
        print(f"  ✓ Performance metrics (accuracy, precision, recall, F1)")
        print(f"  ✓ Drift detection (prediction distribution)")
        print(f"  ✓ Latency monitoring (p95, p99, outliers)")
        print(f"  ✓ Automated alerting ({len(all_alerts)} alerts)")
        print(f"  ✓ Actionable recommendations ({len(recommendations)} items)")
        print("\n")

    return results


# =============================================================================
# DOCUMENTATION
# =============================================================================
"""
USAGE EXAMPLES FOR TEAM:
=========================

Example 1: Quick demo for presentation
---------------------------------------
from observability import monitoring

# Run complete demo (perfect for class presentation!)
results = monitoring.run_monitoring_demo(verbose=True)

# Access specific components
print(results['baseline_report']['summary'])


Example 2: Evaluate real model
-------------------------------
from observability import monitoring

# After training your model
y_true, y_pred, y_prob = get_test_predictions(model, test_data)

# Comprehensive evaluation
report = monitoring.evaluate_batch(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    batch_name="final_test_evaluation",
    reference_predictions=val_predictions  # For drift check
)

# Print summary
print(report['summary'])

# Check alerts
if report['alerts']:
    for alert in report['alerts']:
        print(alert)


Example 3: Latency monitoring
------------------------------
from observability import monitoring
import time

# Collect latency during inference
latencies = []
for batch in test_batches:
    start = time.time()
    predictions = model.predict(batch)
    latency_ms = (time.time() - start) * 1000
    latencies.append(latency_ms)

# Analyze latency
latencies = np.array(latencies)
stats = monitoring.compute_latency_stats(latencies)

print(f"p95 latency: {stats['p95']:.1f}ms")

# Check for latency alerts
from observability import alerts
latency_alerts = alerts.check_latency_alerts(stats)
if latency_alerts:
    print("Latency alerts detected!")


Example 4: Continuous monitoring pipeline
------------------------------------------
from observability import monitoring, mlflow_config
import mlflow

mlflow_config.setup_mlflow()

# Weekly monitoring
for week in range(1, 13):
    mlflow_config.start_run(run_name=f"monitoring_week_{week}")

    # Get week's predictions
    y_true, y_pred, y_prob = get_weekly_predictions(week)

    # Evaluate
    report = monitoring.evaluate_batch(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        batch_name=f"week_{week}",
        reference_predictions=baseline_predictions
    )

    # Log to MLflow
    mlflow.log_metrics(report['metrics'])
    mlflow.log_metric('num_alerts', len(report['alerts']))

    # Handle alerts
    if any(a.severity in ['critical', 'high'] for a in report['alerts']):
        send_alert_notification(report)

    mlflow_config.end_run()
"""
