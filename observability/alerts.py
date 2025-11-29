"""
Alerting Module

ROLE IN OBSERVABILITY LAYER:
-----------------------------
This module implements automated alerting based on configurable thresholds.
It supports:
- MLO 8.7: Production monitoring with actionable alerts
- MLO 14.1: Demonstrating proactive monitoring in final presentation
- Clinical safety: Catching dangerous model behaviors before harm occurs

WHY ALERTS MATTER:
------------------
In production ML systems, especially safety-critical applications like medical
imaging, automated alerts enable:

1. EARLY WARNING SYSTEM:
   - Detect model degradation before it affects many patients
   - Catch data quality issues immediately
   - Identify drift before model becomes unreliable

2. CLINICAL SAFETY:
   - Alert if recall drops (missing tumors!)
   - Alert if precision drops (unnecessary procedures)
   - Alert if model latency spikes (deployment issue)

3. OPERATIONAL MONITORING:
   - Track model health in real-time
   - Enable quick response to issues
   - Maintain audit trail of alerts

ALERT TYPES IN THIS MODULE:
----------------------------
1. Performance alerts: Metrics below threshold (recall, precision, F1)
2. Drift alerts: Data or prediction distribution changes
3. Data quality alerts: Corrupted images, brightness issues
4. Operational alerts: High latency, failures

ARCHITECTURE-AGNOSTIC:
----------------------
All alerts are based on generic metrics and behaviors, not model architecture.
Works with any model that provides predictions and standard metrics.

PRODUCTION INTEGRATION:
-----------------------
In a real deployment, alerts would be sent via:
- Email notifications
- Slack/Teams messages
- PagerDuty for critical issues
- Dashboard warning indicators
- MLflow logging for audit trail

For this course project, we focus on the alert *logic* and *demonstration*.
The delivery mechanism can be added later.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json


# =============================================================================
# ALERT CONFIGURATION
# =============================================================================

class AlertConfig:
    """
    Configuration for alert thresholds.

    This class stores all configurable thresholds for different alert types.
    Teams can adjust these based on their risk tolerance and clinical requirements.
    """

    def __init__(self):
        # PERFORMANCE THRESHOLDS
        # These are based on project SMART goals: >= 80% precision and recall
        self.min_recall = 0.80          # CRITICAL: Below this = missing tumors!
        self.min_precision = 0.80       # IMPORTANT: Below this = too many false positives
        self.min_f1 = 0.75              # Overall performance threshold
        self.min_accuracy = 0.75        # Minimum acceptable accuracy

        # DRIFT THRESHOLDS
        self.max_prediction_drift = 0.30   # Max acceptable change in positive rate
        self.max_embedding_drift = 0.30    # Max acceptable embedding distance

        # DATA QUALITY THRESHOLDS
        self.max_corruption_rate = 0.05    # Max 5% corrupted images acceptable
        self.min_brightness = 20           # Images darker than this are suspicious
        self.max_brightness = 240          # Images brighter than this are suspicious

        # OPERATIONAL THRESHOLDS
        self.max_latency_ms = 1000         # Max acceptable inference latency (1 second)
        self.max_latency_p95_ms = 2000     # 95th percentile latency threshold

        # ALERT SEVERITIES
        # How much below threshold triggers different severity levels
        self.severity_critical_factor = 0.1   # >10% below = critical
        self.severity_high_factor = 0.05      # 5-10% below = high
        self.severity_medium_factor = 0.02    # 2-5% below = medium

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return {
            'performance_thresholds': {
                'min_recall': self.min_recall,
                'min_precision': self.min_precision,
                'min_f1': self.min_f1,
                'min_accuracy': self.min_accuracy,
            },
            'drift_thresholds': {
                'max_prediction_drift': self.max_prediction_drift,
                'max_embedding_drift': self.max_embedding_drift,
            },
            'data_quality_thresholds': {
                'max_corruption_rate': self.max_corruption_rate,
                'min_brightness': self.min_brightness,
                'max_brightness': self.max_brightness,
            },
            'operational_thresholds': {
                'max_latency_ms': self.max_latency_ms,
                'max_latency_p95_ms': self.max_latency_p95_ms,
            }
        }


# Default configuration instance
DEFAULT_CONFIG = AlertConfig()


# =============================================================================
# ALERT DATA STRUCTURES
# =============================================================================

class Alert:
    """
    Represents a single alert event.

    Attributes:
        alert_type (str): Type of alert ('performance', 'drift', 'quality', 'operational')
        severity (str): Severity level ('info', 'warning', 'high', 'critical')
        message (str): Human-readable alert message
        metric_name (str): Name of metric that triggered alert
        metric_value (float): Current value of metric
        threshold (float): Threshold that was violated
        timestamp (str): ISO format timestamp
        metadata (dict): Additional context
    """

    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metric_name: str,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.threshold = threshold
        self.timestamp = datetime.now().isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }

    def __str__(self) -> str:
        """String representation for printing."""
        severity_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'high': '🔴',
            'critical': '🚨'
        }
        emoji = severity_emoji.get(self.severity, '•')

        s = f"{emoji} [{self.severity.upper()}] {self.alert_type.upper()}: {self.message}"
        if self.metric_value is not None and self.threshold is not None:
            s += f" ({self.metric_name}={self.metric_value:.3f}, threshold={self.threshold:.3f})"
        return s


# =============================================================================
# ALERT CHECKING FUNCTIONS
# =============================================================================

def check_performance_alerts(
    metrics: Dict[str, float],
    config: AlertConfig = DEFAULT_CONFIG
) -> List[Alert]:
    """
    Check performance metrics and generate alerts if thresholds violated.

    Args:
        metrics (dict): Metrics from metrics.compute_classification_metrics()
        config (AlertConfig): Alert configuration with thresholds

    Returns:
        list: List of Alert objects (empty if all metrics OK)

    Example:
        >>> from observability import metrics, alerts
        >>> results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)
        >>> alert_list = alerts.check_performance_alerts(results)
        >>> for alert in alert_list:
        >>>     print(alert)
    """
    alert_list = []

    # RECALL CHECK (MOST CRITICAL)
    # Low recall = missing tumors = dangerous!
    recall = metrics.get('recall', 1.0)
    if recall < config.min_recall:
        deficit = config.min_recall - recall

        if deficit > config.severity_critical_factor:
            severity = 'critical'
        elif deficit > config.severity_high_factor:
            severity = 'high'
        else:
            severity = 'warning'

        alert_list.append(Alert(
            alert_type='performance',
            severity=severity,
            message=f"Recall below threshold! Missing {(1-recall)*100:.1f}% of tumors. "
                   f"This is clinically dangerous!",
            metric_name='recall',
            metric_value=recall,
            threshold=config.min_recall,
            metadata={
                'false_negatives': metrics.get('false_negatives', 'unknown'),
                'support_positive': metrics.get('support_positive', 'unknown')
            }
        ))

    # PRECISION CHECK
    # Low precision = too many false positives
    precision = metrics.get('precision', 1.0)
    if precision < config.min_precision:
        deficit = config.min_precision - precision

        if deficit > config.severity_critical_factor:
            severity = 'high'
        elif deficit > config.severity_high_factor:
            severity = 'warning'
        else:
            severity = 'info'

        alert_list.append(Alert(
            alert_type='performance',
            severity=severity,
            message=f"Precision below threshold. {(1-precision)*100:.1f}% of tumor predictions "
                   f"are false positives.",
            metric_name='precision',
            metric_value=precision,
            threshold=config.min_precision,
            metadata={
                'false_positives': metrics.get('false_positives', 'unknown')
            }
        ))

    # F1 SCORE CHECK
    f1 = metrics.get('f1', 1.0)
    if f1 < config.min_f1:
        alert_list.append(Alert(
            alert_type='performance',
            severity='warning',
            message=f"F1 score below threshold. Model performance degraded.",
            metric_name='f1',
            metric_value=f1,
            threshold=config.min_f1
        ))

    # ACCURACY CHECK (less critical than recall/precision for medical)
    accuracy = metrics.get('accuracy', 1.0)
    if accuracy < config.min_accuracy:
        alert_list.append(Alert(
            alert_type='performance',
            severity='info',
            message=f"Overall accuracy below threshold.",
            metric_name='accuracy',
            metric_value=accuracy,
            threshold=config.min_accuracy
        ))

    return alert_list


def check_drift_alerts(
    drift_report: Dict[str, Any],
    config: AlertConfig = DEFAULT_CONFIG
) -> List[Alert]:
    """
    Check drift detection results and generate alerts.

    Args:
        drift_report (dict): Report from drift.monitor_drift()
        config (AlertConfig): Alert configuration

    Returns:
        list: List of Alert objects

    Example:
        >>> from observability import drift, alerts
        >>> drift_report = drift.monitor_drift(ref_preds, cur_preds)
        >>> alert_list = alerts.check_drift_alerts(drift_report)
    """
    alert_list = []

    # Check prediction drift
    if 'prediction_drift' in drift_report:
        pred_drift = drift_report['prediction_drift']

        if pred_drift['drift_detected']:
            # Map severity from drift module to alert severity
            severity_map = {
                'low': 'info',
                'medium': 'warning',
                'high': 'high'
            }
            severity = severity_map.get(pred_drift['severity'], 'warning')

            rate_change = pred_drift['current_positive_rate'] - pred_drift['reference_positive_rate']
            direction = "increased" if rate_change > 0 else "decreased"

            alert_list.append(Alert(
                alert_type='drift',
                severity=severity,
                message=f"Prediction drift detected! Positive prediction rate {direction} "
                       f"from {pred_drift['reference_positive_rate']:.2%} to "
                       f"{pred_drift['current_positive_rate']:.2%}.",
                metric_name='prediction_drift',
                metric_value=pred_drift['positive_rate_diff'],
                threshold=config.max_prediction_drift,
                metadata={
                    'chi2_pvalue': pred_drift.get('chi2_pvalue'),
                    'reference_size': pred_drift['reference_size'],
                    'current_size': pred_drift['current_size']
                }
            ))

    # Check embedding drift (if available)
    if drift_report.get('embedding_drift') is not None:
        emb_drift = drift_report['embedding_drift']

        if emb_drift['drift_detected']:
            severity_map = {
                'low': 'info',
                'medium': 'warning',
                'high': 'high'
            }
            severity = severity_map.get(emb_drift['severity'], 'warning')

            alert_list.append(Alert(
                alert_type='drift',
                severity=severity,
                message=f"Embedding drift detected! Input data characteristics have changed.",
                metric_name='embedding_drift',
                metric_value=emb_drift['drift_score'],
                threshold=config.max_embedding_drift,
                metadata={
                    'method': emb_drift['method']
                }
            ))

    return alert_list


def check_data_quality_alerts(
    quality_report: Dict[str, Any],
    config: AlertConfig = DEFAULT_CONFIG
) -> List[Alert]:
    """
    Check data quality report and generate alerts.

    Args:
        quality_report (dict): Report from data_quality.analyze_dataset_quality()
        config (AlertConfig): Alert configuration

    Returns:
        list: List of Alert objects

    Example:
        >>> from observability import data_quality, alerts
        >>> report = data_quality.analyze_dataset_quality(image_paths)
        >>> alert_list = alerts.check_data_quality_alerts(report)
    """
    alert_list = []

    # Check corruption rate
    corruption_rate = quality_report.get('corrupted_percentage', 0) / 100

    if corruption_rate > config.max_corruption_rate:
        if corruption_rate > config.max_corruption_rate * 3:
            severity = 'critical'
        elif corruption_rate > config.max_corruption_rate * 2:
            severity = 'high'
        else:
            severity = 'warning'

        alert_list.append(Alert(
            alert_type='quality',
            severity=severity,
            message=f"High image corruption rate: {corruption_rate*100:.1f}% of images "
                   f"are corrupted or invalid. Check data pipeline!",
            metric_name='corruption_rate',
            metric_value=corruption_rate,
            threshold=config.max_corruption_rate,
            metadata={
                'corrupted_count': quality_report.get('corrupted_count'),
                'total_images': quality_report.get('total_images')
            }
        ))

    # Check brightness extremes
    if quality_report.get('brightness_mean') is not None:
        mean_brightness = quality_report['brightness_mean']

        if mean_brightness < config.min_brightness:
            alert_list.append(Alert(
                alert_type='quality',
                severity='warning',
                message=f"Images unusually dark (mean brightness={mean_brightness:.1f}). "
                       f"Check scanner calibration or preprocessing.",
                metric_name='brightness_mean',
                metric_value=mean_brightness,
                threshold=config.min_brightness
            ))

        if mean_brightness > config.max_brightness:
            alert_list.append(Alert(
                alert_type='quality',
                severity='warning',
                message=f"Images unusually bright (mean brightness={mean_brightness:.1f}). "
                       f"Check scanner calibration or preprocessing.",
                metric_name='brightness_mean',
                metric_value=mean_brightness,
                threshold=config.max_brightness
            ))

    return alert_list


def check_latency_alerts(
    latency_stats: Dict[str, float],
    config: AlertConfig = DEFAULT_CONFIG
) -> List[Alert]:
    """
    Check inference latency and generate alerts if too slow.

    Args:
        latency_stats (dict): Latency statistics with keys like 'mean', 'p95', 'max'
        config (AlertConfig): Alert configuration

    Returns:
        list: List of Alert objects

    Example:
        >>> latency_stats = {
        >>>     'mean': 450,      # ms
        >>>     'p95': 1200,      # ms
        >>>     'max': 2500       # ms
        >>> }
        >>> alert_list = alerts.check_latency_alerts(latency_stats)
    """
    alert_list = []

    # Check 95th percentile latency (more robust than max)
    p95_latency = latency_stats.get('p95', 0)

    if p95_latency > config.max_latency_p95_ms:
        if p95_latency > config.max_latency_p95_ms * 2:
            severity = 'high'
        else:
            severity = 'warning'

        alert_list.append(Alert(
            alert_type='operational',
            severity=severity,
            message=f"High inference latency: p95={p95_latency:.0f}ms. "
                   f"Model may be overloaded or deployment issue.",
            metric_name='latency_p95',
            metric_value=p95_latency,
            threshold=config.max_latency_p95_ms,
            metadata=latency_stats
        ))

    # Check mean latency
    mean_latency = latency_stats.get('mean', 0)
    if mean_latency > config.max_latency_ms:
        alert_list.append(Alert(
            alert_type='operational',
            severity='info',
            message=f"Mean inference latency elevated: {mean_latency:.0f}ms",
            metric_name='latency_mean',
            metric_value=mean_latency,
            threshold=config.max_latency_ms
        ))

    return alert_list


# =============================================================================
# UNIFIED ALERT CHECKING
# =============================================================================

def check_all_alerts(
    metrics: Optional[Dict[str, float]] = None,
    drift_report: Optional[Dict[str, Any]] = None,
    quality_report: Optional[Dict[str, Any]] = None,
    latency_stats: Optional[Dict[str, float]] = None,
    config: AlertConfig = DEFAULT_CONFIG
) -> List[Alert]:
    """
    Check all alert types at once and return comprehensive alert list.

    This is the main function to use for production monitoring.

    Args:
        metrics (dict, optional): Performance metrics
        drift_report (dict, optional): Drift monitoring report
        quality_report (dict, optional): Data quality report
        latency_stats (dict, optional): Latency statistics
        config (AlertConfig): Alert configuration

    Returns:
        list: All alerts from all checks

    Example:
        >>> from observability import metrics, drift, data_quality, alerts
        >>>
        >>> # Compute all monitoring data
        >>> perf_metrics = metrics.compute_classification_metrics(y_true, y_pred, y_prob)
        >>> drift_report = drift.monitor_drift(ref_preds, cur_preds)
        >>> quality_report = data_quality.analyze_dataset_quality(images)
        >>> latency_stats = {'mean': 450, 'p95': 800}
        >>>
        >>> # Check all alerts
        >>> all_alerts = alerts.check_all_alerts(
        >>>     metrics=perf_metrics,
        >>>     drift_report=drift_report,
        >>>     quality_report=quality_report,
        >>>     latency_stats=latency_stats
        >>> )
        >>>
        >>> # Print and handle alerts
        >>> for alert in all_alerts:
        >>>     print(alert)
        >>>     if alert.severity in ['critical', 'high']:
        >>>         send_urgent_notification(alert)
    """
    all_alerts = []

    if metrics is not None:
        all_alerts.extend(check_performance_alerts(metrics, config))

    if drift_report is not None:
        all_alerts.extend(check_drift_alerts(drift_report, config))

    if quality_report is not None:
        all_alerts.extend(check_data_quality_alerts(quality_report, config))

    if latency_stats is not None:
        all_alerts.extend(check_latency_alerts(latency_stats, config))

    return all_alerts


def print_alerts(alerts: List[Alert], show_metadata: bool = False) -> None:
    """
    Pretty-print a list of alerts.

    Args:
        alerts (list): List of Alert objects
        show_metadata (bool): Whether to show detailed metadata

    Example:
        >>> alerts_list = check_all_alerts(metrics=perf_metrics)
        >>> print_alerts(alerts_list)
    """
    if not alerts:
        print("\n✓ No alerts - all systems normal\n")
        return

    print("\n" + "=" * 70)
    print(f"ALERT SUMMARY - {len(alerts)} alert(s)")
    print("=" * 70)

    # Group by severity
    by_severity = {'critical': [], 'high': [], 'warning': [], 'info': []}
    for alert in alerts:
        by_severity[alert.severity].append(alert)

    # Print in order of severity
    for severity in ['critical', 'high', 'warning', 'info']:
        severity_alerts = by_severity[severity]
        if severity_alerts:
            print(f"\n{severity.upper()} ({len(severity_alerts)}):")
            for alert in severity_alerts:
                print(f"  {alert}")
                if show_metadata and alert.metadata:
                    print(f"    Metadata: {alert.metadata}")

    print("=" * 70 + "\n")


def save_alerts(alerts: List[Alert], filepath: str) -> None:
    """
    Save alerts to JSON file for audit trail.

    Args:
        alerts (list): List of Alert objects
        filepath (str): Path to save JSON file

    Example:
        >>> alerts_list = check_all_alerts(metrics=perf_metrics)
        >>> save_alerts(alerts_list, "alerts_2025_01_15.json")
    """
    alert_dicts = [alert.to_dict() for alert in alerts]

    with open(filepath, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_alerts': len(alerts),
            'alerts': alert_dicts
        }, f, indent=2)

    print(f"[Alerts] Saved {len(alerts)} alerts to {filepath}")


# =============================================================================
# DOCUMENTATION
# =============================================================================
"""
USAGE EXAMPLES FOR TEAM:
=========================

Example 1: Basic alert checking
--------------------------------
from observability import metrics, alerts

# Compute metrics
results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)

# Check for performance alerts
alert_list = alerts.check_performance_alerts(results)

# Print alerts
alerts.print_alerts(alert_list)


Example 2: Comprehensive monitoring
------------------------------------
from observability import metrics, drift, data_quality, alerts

# Gather all monitoring data
perf_metrics = metrics.compute_classification_metrics(y_true, y_pred, y_prob)
drift_report = drift.monitor_drift(baseline_preds, current_preds)
quality_report = data_quality.analyze_dataset_quality(image_paths)
latency_stats = {'mean': 500, 'p95': 900, 'max': 1500}

# Check all alerts
all_alerts = alerts.check_all_alerts(
    metrics=perf_metrics,
    drift_report=drift_report,
    quality_report=quality_report,
    latency_stats=latency_stats
)

# Handle alerts
alerts.print_alerts(all_alerts)

# Save for audit trail
alerts.save_alerts(all_alerts, f"alerts_{date}.json")

# Take action on critical alerts
critical_alerts = [a for a in all_alerts if a.severity == 'critical']
if critical_alerts:
    send_urgent_notification(critical_alerts)


Example 3: Custom thresholds
-----------------------------
from observability import alerts

# Create custom configuration (more strict)
custom_config = alerts.AlertConfig()
custom_config.min_recall = 0.90        # Require 90% recall
custom_config.min_precision = 0.85     # Require 85% precision

# Use custom config
alert_list = alerts.check_performance_alerts(results, config=custom_config)


Example 4: Integration with MLflow
-----------------------------------
from observability import mlflow_config, metrics, alerts
import mlflow

mlflow_config.setup_mlflow()
mlflow_config.start_run(run_name="weekly_monitoring")

# Compute metrics
results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)

# Check alerts
alert_list = alerts.check_performance_alerts(results)

# Log alert count to MLflow
mlflow.log_metric("num_alerts", len(alert_list))
mlflow.log_metric("num_critical_alerts",
                  sum(1 for a in alert_list if a.severity == 'critical'))

# Log alerts as parameter
if alert_list:
    mlflow.log_param("has_alerts", True)
    mlflow.log_param("alert_summary", f"{len(alert_list)} alerts")

mlflow_config.end_run()
"""
