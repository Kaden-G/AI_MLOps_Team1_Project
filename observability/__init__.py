"""
Observability Package for Brain Tumor Classification MLOps Pipeline

This package provides a complete, model-agnostic observability layer for monitoring
and tracking machine learning models in the Johns Hopkins 635.603 AI/ML Ops course project.

PURPOSE:
--------
Support MLO objectives 8.1, 8.4, 8.5, 8.6, 8.7, and 14.1 by providing:
- Experiment tracking via MLflow
- Comprehensive evaluation metrics
- Data quality monitoring
- Model drift detection
- Performance monitoring
- Automated alerting

ARCHITECTURE-AGNOSTIC DESIGN:
------------------------------
All modules in this package are designed to work with ANY CNN architecture:
- Basic CNNs
- Custom CNNs
- Transfer learning models (ResNet, EfficientNet, etc.)
- Future architectures

The only requirements are standard outputs:
- y_true: Ground truth labels (binary: 0 or 1)
- y_pred: Predicted labels (binary: 0 or 1)
- y_prob: Prediction probabilities (optional, shape: (n_samples, n_classes) or (n_samples,))
- embeddings: Feature vectors (optional, for drift detection)

MODULES:
--------
- mlflow_config: MLflow experiment tracking setup
- logger: Logging utilities for training and inference
- metrics: Model evaluation metrics (precision, recall, F1, ROC-AUC)
- data_quality: Image quality checks (brightness, corruption detection)
- drift: Prediction and embedding drift detection
- alerts: Automated alerting based on configurable thresholds
- monitoring: End-to-end monitoring utilities

USAGE:
------
    from observability import mlflow_config, metrics, alerts

    # Setup MLflow
    mlflow_config.setup_mlflow()

    # Compute metrics
    results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)

    # Check for alerts
    alert_list = alerts.check_alerts(results)

AUTHORS: Johns Hopkins 635.603 Team (Fall 2025)
PROJECT: Brain MRI Tumor Classification
"""

__version__ = "1.0.0"
__author__ = "JHU 635.603 MLOps Team"

# Make key functions available at package level for convenience
from .mlflow_config import setup_mlflow, start_run, end_run
from .metrics import compute_classification_metrics, print_metrics
from .alerts import check_alerts
from .monitoring import run_monitoring_demo

__all__ = [
    'setup_mlflow',
    'start_run',
    'end_run',
    'compute_classification_metrics',
    'print_metrics',
    'check_alerts',
    'run_monitoring_demo',
]
