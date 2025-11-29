"""
Logging Module for Training and Inference Events

ROLE IN OBSERVABILITY LAYER:
-----------------------------
This module provides standardized logging functions for both training and inference phases.
It supports MLO 8.4 (tracking/metadata) and MLO 8.5 (production-grade code) by ensuring
consistent, structured logging of all model behaviors.

WHY SEPARATE FROM MLFLOW_CONFIG?
---------------------------------
While mlflow_config.py provides the *infrastructure* for logging, this module provides
*convenience functions* that make logging easier and more consistent across the team.
It wraps MLflow calls with domain-specific logic for our brain tumor classification task.

ARCHITECTURE-AGNOSTIC DESIGN:
------------------------------
All functions accept generic inputs (losses, accuracies, predictions) without any
assumptions about the underlying model architecture. Whether your model is a basic CNN,
ResNet, or custom architecture, these functions work the same way.

WHAT TO LOG:
------------
Training Phase:
- Hyperparameters (learning rate, batch size, optimizer, etc.)
- Per-epoch metrics (loss, accuracy, precision, recall, F1)
- Model architecture summary (as text, not code-dependent)

Inference Phase:
- Prediction latency
- Prediction confidence scores
- Model version used
- Batch size
- Any anomalies or errors
"""

import mlflow
import time
from typing import Dict, Any, Optional, List
from datetime import datetime


# =============================================================================
# TRAINING LOGGING FUNCTIONS
# =============================================================================

def log_training_params(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    optimizer: str,
    loss_function: str,
    architecture: str = "cnn",
    additional_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log training hyperparameters to MLflow.

    This function logs the core training configuration that affects model performance.
    These parameters are crucial for reproducibility and comparing experiments.

    Args:
        learning_rate (float): Learning rate used for training (e.g., 0.001).
        batch_size (int): Number of samples per batch (e.g., 32).
        epochs (int): Total number of training epochs (e.g., 50).
        optimizer (str): Optimizer name (e.g., "adam", "sgd", "rmsprop").
        loss_function (str): Loss function name (e.g., "binary_crossentropy", "focal_loss").
        architecture (str): High-level architecture description (e.g., "custom_cnn", "resnet50").
            This should be descriptive but NOT architecture-specific code.
        additional_params (dict, optional): Any additional parameters specific to your model.
            Examples: {"dropout_rate": 0.5, "l2_regularization": 0.01}

    Example:
        >>> from observability import mlflow_config, logger
        >>> mlflow_config.setup_mlflow()
        >>> mlflow_config.start_run(run_name="baseline_experiment")
        >>>
        >>> logger.log_training_params(
        >>>     learning_rate=0.001,
        >>>     batch_size=32,
        >>>     epochs=50,
        >>>     optimizer="adam",
        >>>     loss_function="binary_crossentropy",
        >>>     architecture="custom_cnn_5_layers",
        >>>     additional_params={"dropout": 0.3, "data_augmentation": True}
        >>> )
        >>>
        >>> # ... training code ...
        >>> mlflow_config.end_run()

    Note:
        Must be called within an active MLflow run (after start_run()).
    """
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run. Call mlflow_config.start_run() first!")

    # Core parameters that every model should have
    params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": optimizer,
        "loss_function": loss_function,
        "architecture": architecture,
        "timestamp": datetime.now().isoformat(),  # When training started
    }

    # Add any additional parameters provided
    if additional_params:
        params.update(additional_params)

    # Log all parameters at once
    for key, value in params.items():
        mlflow.log_param(key, value)

    print(f"[Logger] Logged {len(params)} training parameters")


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_accuracy: Optional[float] = None,
    val_loss: Optional[float] = None,
    val_accuracy: Optional[float] = None,
    val_precision: Optional[float] = None,
    val_recall: Optional[float] = None,
    val_f1: Optional[float] = None,
    additional_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log metrics for a single training epoch.

    This function should be called at the end of each epoch to track model
    performance over time. MLflow will plot these metrics automatically.

    Args:
        epoch (int): Current epoch number (0-indexed or 1-indexed, your choice).
        train_loss (float): Training loss for this epoch.
        train_accuracy (float, optional): Training accuracy for this epoch.
        val_loss (float, optional): Validation loss for this epoch.
        val_accuracy (float, optional): Validation accuracy for this epoch.
        val_precision (float, optional): Validation precision for this epoch.
            CLINICALLY IMPORTANT for brain tumor detection (minimize false negatives).
        val_recall (float, optional): Validation recall for this epoch.
            CLINICALLY CRITICAL - recall < 0.80 means missing tumor cases!
        val_f1 (float, optional): Validation F1 score for this epoch.
        additional_metrics (dict, optional): Any other metrics to log
            (e.g., {"auc": 0.95, "specificity": 0.88}).

    Example:
        >>> for epoch in range(50):
        >>>     # ... training code ...
        >>>     train_loss, train_acc = train_one_epoch()
        >>>     val_loss, val_acc, val_prec, val_rec = validate()
        >>>
        >>>     logger.log_epoch_metrics(
        >>>         epoch=epoch,
        >>>         train_loss=train_loss,
        >>>         train_accuracy=train_acc,
        >>>         val_loss=val_loss,
        >>>         val_accuracy=val_acc,
        >>>         val_precision=val_prec,
        >>>         val_recall=val_rec,
        >>>         val_f1=2 * val_prec * val_rec / (val_prec + val_rec)
        >>>     )

    Note:
        The 'epoch' parameter is used as the 'step' in MLflow, allowing you to
        see how metrics evolve across training.
    """
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run. Call mlflow_config.start_run() first!")

    # Build metrics dictionary with only non-None values
    metrics = {"train_loss": train_loss}

    if train_accuracy is not None:
        metrics["train_accuracy"] = train_accuracy
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if val_accuracy is not None:
        metrics["val_accuracy"] = val_accuracy
    if val_precision is not None:
        metrics["val_precision"] = val_precision
    if val_recall is not None:
        metrics["val_recall"] = val_recall
    if val_f1 is not None:
        metrics["val_f1"] = val_f1

    # Add any additional metrics
    if additional_metrics:
        metrics.update(additional_metrics)

    # Log all metrics for this epoch
    mlflow.log_metrics(metrics, step=epoch)

    # Pretty print for console feedback
    print(f"[Logger] Epoch {epoch}: {_format_metrics(metrics)}")


def log_training_summary(
    total_epochs_completed: int,
    best_val_accuracy: float,
    best_val_recall: float,
    best_val_precision: float,
    final_train_loss: float,
    training_time_seconds: float,
    additional_summary: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a summary of the entire training run.

    Call this once at the end of training to record overall statistics.
    These are separate from epoch-by-epoch metrics and provide a high-level view.

    Args:
        total_epochs_completed (int): How many epochs were completed.
        best_val_accuracy (float): Best validation accuracy achieved.
        best_val_recall (float): Best validation recall achieved.
            CRITICAL: This should be >= 0.80 for clinical safety!
        best_val_precision (float): Best validation precision achieved.
        final_train_loss (float): Training loss at the end of training.
        training_time_seconds (float): Total time spent training (in seconds).
        additional_summary (dict, optional): Any additional summary statistics.

    Example:
        >>> start_time = time.time()
        >>> # ... training loop ...
        >>> end_time = time.time()
        >>>
        >>> logger.log_training_summary(
        >>>     total_epochs_completed=50,
        >>>     best_val_accuracy=0.92,
        >>>     best_val_recall=0.89,
        >>>     best_val_precision=0.91,
        >>>     final_train_loss=0.15,
        >>>     training_time_seconds=end_time - start_time,
        >>>     additional_summary={"early_stopped": True, "stopped_at_epoch": 35}
        >>> )
    """
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run. Call mlflow_config.start_run() first!")

    # Log summary as both metrics (for visualization) and params (for searchability)
    summary_metrics = {
        "best_val_accuracy": best_val_accuracy,
        "best_val_recall": best_val_recall,
        "best_val_precision": best_val_precision,
        "final_train_loss": final_train_loss,
        "training_time_seconds": training_time_seconds,
    }

    summary_params = {
        "total_epochs_completed": total_epochs_completed,
        "training_time_minutes": round(training_time_seconds / 60, 2),
    }

    # Add additional summary items
    if additional_summary:
        for key, value in additional_summary.items():
            if isinstance(value, (int, float)):
                summary_metrics[key] = value
            else:
                summary_params[key] = str(value)

    # Log everything
    mlflow.log_metrics(summary_metrics)
    for key, value in summary_params.items():
        mlflow.log_param(key, value)

    print(f"[Logger] Training summary logged:")
    print(f"  - Best validation recall: {best_val_recall:.4f}")
    print(f"  - Best validation precision: {best_val_precision:.4f}")
    print(f"  - Training time: {training_time_seconds / 60:.2f} minutes")


# =============================================================================
# INFERENCE LOGGING FUNCTIONS
# =============================================================================

def log_inference_batch(
    predictions: List[int],
    prediction_probabilities: Optional[List[float]] = None,
    latency_ms: Optional[float] = None,
    model_version: str = "unknown",
    batch_id: Optional[str] = None
) -> None:
    """
    Log inference results for a batch of predictions.

    In production or during evaluation, this function tracks model behavior
    on new data, which is critical for monitoring model health.

    Args:
        predictions (list of int): Binary predictions (0 or 1).
        prediction_probabilities (list of float, optional): Confidence scores
            for positive class (tumor detected). Values should be in [0, 1].
        latency_ms (float, optional): Time taken to generate predictions (milliseconds).
        model_version (str): Identifier for which model was used (e.g., "v1", "baseline_cnn").
        batch_id (str, optional): Unique identifier for this batch (for tracking).

    Example:
        >>> # During inference
        >>> start = time.time()
        >>> predictions = model.predict(test_images)  # Returns probabilities
        >>> y_pred = (predictions > 0.5).astype(int)  # Convert to binary
        >>> latency = (time.time() - start) * 1000  # Convert to ms
        >>>
        >>> logger.log_inference_batch(
        >>>     predictions=y_pred.tolist(),
        >>>     prediction_probabilities=predictions[:, 1].tolist(),
        >>>     latency_ms=latency,
        >>>     model_version="baseline_cnn_v1",
        >>>     batch_id="test_batch_001"
        >>> )

    Note:
        This is architecture-agnostic! Works with any model's output.
    """
    if not mlflow.active_run():
        # Inference logging often happens outside of training runs
        # We can either warn or auto-create a run
        print("[Logger] Warning: No active MLflow run for inference logging")
        return

    # Compute summary statistics for this batch
    num_predictions = len(predictions)
    num_positive_predictions = sum(predictions)  # How many "tumor" predictions
    positive_rate = num_positive_predictions / num_predictions if num_predictions > 0 else 0

    # Average confidence if provided
    avg_confidence = None
    if prediction_probabilities:
        avg_confidence = sum(prediction_probabilities) / len(prediction_probabilities)

    # Log batch-level metrics
    batch_metrics = {
        "inference_batch_size": num_predictions,
        "inference_positive_rate": positive_rate,
        "inference_num_positive": num_positive_predictions,
    }

    if latency_ms is not None:
        batch_metrics["inference_latency_ms"] = latency_ms
        batch_metrics["inference_latency_per_sample_ms"] = latency_ms / num_predictions

    if avg_confidence is not None:
        batch_metrics["inference_avg_confidence"] = avg_confidence

    # Log as metrics (can track over time if doing multiple inference batches)
    mlflow.log_metrics(batch_metrics)

    # Log metadata as params
    mlflow.log_param("model_version", model_version)
    if batch_id:
        mlflow.log_param("batch_id", batch_id)

    print(f"[Logger] Inference batch logged: {num_predictions} predictions, "
          f"{positive_rate:.2%} positive, latency: {latency_ms:.2f}ms")


def log_inference_anomaly(
    anomaly_type: str,
    description: str,
    severity: str = "warning",
    sample_id: Optional[str] = None
) -> None:
    """
    Log unusual or concerning events during inference.

    Examples of anomalies:
    - Very low confidence predictions
    - Unexpected input data (corrupted images, wrong dimensions)
    - Latency spikes
    - Model returning NaN or invalid outputs

    Args:
        anomaly_type (str): Category of anomaly (e.g., "low_confidence", "corrupted_input").
        description (str): Human-readable description of what happened.
        severity (str): "info", "warning", or "critical".
        sample_id (str, optional): Identifier for the specific sample (if applicable).

    Example:
        >>> if confidence < 0.3:  # Very uncertain prediction
        >>>     logger.log_inference_anomaly(
        >>>         anomaly_type="low_confidence",
        >>>         description=f"Prediction confidence only {confidence:.2f}",
        >>>         severity="warning",
        >>>         sample_id="patient_12345"
        >>>     )
    """
    timestamp = datetime.now().isoformat()

    # Log as a parameter (searchable)
    anomaly_key = f"anomaly_{timestamp}"
    anomaly_value = f"[{severity.upper()}] {anomaly_type}: {description}"
    if sample_id:
        anomaly_value += f" (sample: {sample_id})"

    if mlflow.active_run():
        mlflow.log_param(anomaly_key, anomaly_value)

    # Also print to console for immediate visibility
    print(f"[Logger] ANOMALY DETECTED - {anomaly_value}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary as readable string.

    Args:
        metrics (dict): Metric name -> value mapping.

    Returns:
        str: Formatted string like "loss=0.45, accuracy=0.82"
    """
    return ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])


def create_inference_run(model_version: str, dataset_name: str = "test_set") -> None:
    """
    Convenience function to start an MLflow run specifically for inference logging.

    Args:
        model_version (str): Version of model being used for inference.
        dataset_name (str): Name of dataset being evaluated (e.g., "test_set", "production_batch_1").

    Example:
        >>> from observability import mlflow_config, logger
        >>> mlflow_config.setup_mlflow()
        >>>
        >>> logger.create_inference_run(
        >>>     model_version="baseline_cnn_v1",
        >>>     dataset_name="holdout_test_set"
        >>> )
        >>>
        >>> # ... run inference ...
        >>> logger.log_inference_batch(predictions, ...)
        >>>
        >>> mlflow_config.end_run()
    """
    import mlflow
    from observability.mlflow_config import start_run

    run_name = f"inference_{model_version}_{dataset_name}"
    start_run(run_name=run_name)

    # Log initial parameters
    mlflow.log_param("model_version", model_version)
    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_param("run_type", "inference")

    print(f"[Logger] Created inference run: {run_name}")


# =============================================================================
# BEST PRACTICES REMINDER
# =============================================================================
"""
LOGGING BEST PRACTICES FOR TEAM:
==================================

1. ALWAYS log within an active MLflow run:
   - Call mlflow_config.start_run() before logging
   - Call mlflow_config.end_run() when done

2. Log training parameters ONCE per run:
   - Use log_training_params() at the start of training

3. Log epoch metrics EVERY epoch:
   - Use log_epoch_metrics() in your training loop

4. Log training summary ONCE at the end:
   - Use log_training_summary() after training completes

5. For inference/evaluation:
   - Create a separate run with create_inference_run()
   - Log batches with log_inference_batch()

6. Handle errors gracefully:
   - If training fails, call end_run(status="FAILED")
   - This helps track unsuccessful experiments

7. Be consistent with naming:
   - Use descriptive run names
   - Use consistent metric names across experiments
   - This makes comparison easier

8. Architecture-agnostic logging:
   - Never log model-specific details (like "resnet_layer_5_output")
   - Use generic terms ("feature_extractor_output", "final_layer_dim")
   - This ensures code works regardless of architecture changes
"""
