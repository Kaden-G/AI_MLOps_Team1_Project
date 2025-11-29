"""
MLflow Configuration Module

ROLE IN OBSERVABILITY LAYER:
-----------------------------
This module provides the foundational experiment tracking infrastructure using MLflow.
It enables the team to satisfy MLO 8.4 (tracking and metadata) and MLO 8.5 (production-grade code)
by providing a centralized, reproducible way to log experiments, parameters, metrics, and artifacts.

WHY MLFLOW?
-----------
MLflow is industry-standard for ML experiment tracking and provides:
- Automatic versioning of experiments
- Parameter and metric logging
- Model artifact storage
- Easy comparison of runs
- Integration with many ML frameworks
- Local and remote deployment options

ARCHITECTURE-AGNOSTIC:
----------------------
This module makes NO assumptions about model architecture. It simply provides
infrastructure to log whatever parameters, metrics, and artifacts your model generates.

FUTURE MIGRATION:
-----------------
Currently configured for local file storage (./mlruns). To migrate to a remote
MLflow server (e.g., for team collaboration or production):
1. Update MLFLOW_TRACKING_URI to point to remote server (e.g., "http://mlflow-server:5000")
2. Configure authentication if needed
3. No code changes required in other modules!
"""

import mlflow
import os
from typing import Optional, Dict, Any
from pathlib import Path


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Local file-based tracking (stores experiments in ./mlruns directory)
# MIGRATION NOTE: Change this to remote server URL when ready for team collaboration
MLFLOW_TRACKING_URI = "file:./mlruns"

# Default experiment name for the project
# All runs will be grouped under this experiment for easy comparison
DEFAULT_EXPERIMENT_NAME = "brain_mri_cnn_classification"

# Artifact location (where model files, plots, etc. are stored)
# Using local directory by default
DEFAULT_ARTIFACT_LOCATION = "./mlruns/artifacts"


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    artifact_location: Optional[str] = None
) -> str:
    """
    Initialize MLflow tracking with specified or default configuration.

    This function should be called once at the start of your training script
    or notebook to configure where MLflow stores experiment data.

    Args:
        tracking_uri (str, optional): URI where MLflow stores data.
            - Local file: "file:./mlruns" (default)
            - Remote server: "http://mlflow-server:5000"
            - Databricks: "databricks"
        experiment_name (str, optional): Name of the experiment.
            All runs will be grouped under this name.
            Defaults to "brain_mri_cnn_classification".
        artifact_location (str, optional): Where to store artifacts (models, plots).
            Defaults to "./mlruns/artifacts".

    Returns:
        str: The experiment ID created or retrieved.

    Example:
        >>> setup_mlflow()
        >>> # Now you can use start_run() to begin logging
    """
    # Use defaults if not provided
    uri = tracking_uri or MLFLOW_TRACKING_URI
    exp_name = experiment_name or DEFAULT_EXPERIMENT_NAME
    artifact_loc = artifact_location or DEFAULT_ARTIFACT_LOCATION

    # Set the tracking URI - this tells MLflow where to store data
    mlflow.set_tracking_uri(uri)
    print(f"[MLflow] Tracking URI set to: {uri}")

    # Create or get the experiment
    # If experiment exists, this returns its ID; otherwise creates new one
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            # Create new experiment
            experiment_id = mlflow.create_experiment(
                exp_name,
                artifact_location=artifact_loc
            )
            print(f"[MLflow] Created new experiment '{exp_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"[MLflow] Using existing experiment '{exp_name}' with ID: {experiment_id}")

        # Set this as the active experiment
        mlflow.set_experiment(exp_name)

        return experiment_id

    except Exception as e:
        print(f"[MLflow] Error setting up MLflow: {e}")
        raise


def start_run(run_name: Optional[str] = None, nested: bool = False) -> mlflow.ActiveRun:
    """
    Start a new MLflow run to log parameters, metrics, and artifacts.

    A "run" represents a single execution of your training script.
    All metrics and parameters logged after calling this function
    will be associated with this run.

    Args:
        run_name (str, optional): Human-readable name for this run.
            Example: "baseline_cnn_v1", "resnet50_transfer_learning"
        nested (bool): Whether this is a nested run (for hyperparameter sweeps).
            Default is False.

    Returns:
        mlflow.ActiveRun: The active run object. You typically don't need this,
            but it's available for advanced use cases.

    Example:
        >>> start_run(run_name="baseline_cnn_experiment")
        >>> mlflow.log_param("learning_rate", 0.001)
        >>> mlflow.log_metric("accuracy", 0.95)
        >>> end_run()

    Note:
        Always pair with end_run() to properly close the run!
    """
    run = mlflow.start_run(run_name=run_name, nested=nested)
    print(f"[MLflow] Started run: {run.info.run_id}" + (f" ('{run_name}')" if run_name else ""))
    return run


def end_run(status: str = "FINISHED") -> None:
    """
    End the current MLflow run.

    This should be called after you're done logging metrics and parameters
    for a run. It marks the run as complete and finalizes all logged data.

    Args:
        status (str): Final status of the run. Options:
            - "FINISHED": Successful completion (default)
            - "FAILED": Run failed
            - "KILLED": Run was manually terminated

    Example:
        >>> start_run()
        >>> # ... training code ...
        >>> mlflow.log_metric("accuracy", 0.95)
        >>> end_run()  # Mark as successfully finished

        >>> # If training fails:
        >>> try:
        >>>     # ... training code ...
        >>> except Exception:
        >>>     end_run(status="FAILED")

    Note:
        If you don't call this, MLflow will auto-close the run but won't
        mark it as finished, making it harder to track successful vs incomplete runs.
    """
    mlflow.end_run(status=status)
    print(f"[MLflow] Ended run with status: {status}")


def get_current_run_id() -> Optional[str]:
    """
    Get the ID of the currently active MLflow run.

    Returns:
        str or None: Run ID if a run is active, None otherwise.

    Example:
        >>> start_run()
        >>> run_id = get_current_run_id()
        >>> print(f"Current run: {run_id}")
    """
    active_run = mlflow.active_run()
    if active_run:
        return active_run.info.run_id
    return None


def log_model_params(params: Dict[str, Any]) -> None:
    """
    Log model parameters to the current MLflow run.

    Parameters are searchable and comparable across runs, making them
    essential for experiment tracking and reproducibility.

    Args:
        params (dict): Dictionary of parameter names and values.
            Keys should be descriptive strings, values can be numbers or strings.

    Example:
        >>> start_run()
        >>> log_model_params({
        >>>     "learning_rate": 0.001,
        >>>     "batch_size": 32,
        >>>     "optimizer": "adam",
        >>>     "architecture": "custom_cnn",
        >>>     "num_layers": 5
        >>> })
        >>> end_run()

    Note:
        This is model-agnostic! Works with any architecture's parameters.
    """
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run. Call start_run() first!")

    # MLflow expects parameters to be logged individually
    for key, value in params.items():
        mlflow.log_param(key, value)
        print(f"[MLflow] Logged parameter: {key}={value}")


def log_metrics_dict(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log multiple metrics at once to the current MLflow run.

    Metrics are the quantitative results of your model (accuracy, loss, etc.).
    MLflow allows tracking metrics over time (via step parameter).

    Args:
        metrics (dict): Dictionary of metric names and values.
            Keys are metric names, values must be numeric.
        step (int, optional): Training step/epoch number. Useful for tracking
            metrics across epochs. If None, treats as single-point metric.

    Example:
        >>> start_run()
        >>> # Log metrics for epoch 1
        >>> log_metrics_dict({
        >>>     "train_loss": 0.45,
        >>>     "train_accuracy": 0.82,
        >>>     "val_accuracy": 0.79
        >>> }, step=1)
        >>> # Log metrics for epoch 2
        >>> log_metrics_dict({
        >>>     "train_loss": 0.32,
        >>>     "train_accuracy": 0.88,
        >>>     "val_accuracy": 0.85
        >>> }, step=2)
        >>> end_run()

    Note:
        This is completely architecture-agnostic!
    """
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run. Call start_run() first!")

    # Log all metrics at once (more efficient than one-by-one)
    mlflow.log_metrics(metrics, step=step)

    # Print confirmation
    step_str = f" (step {step})" if step is not None else ""
    print(f"[MLflow] Logged {len(metrics)} metrics{step_str}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_mlflow_ui_command() -> str:
    """
    Get the command to launch the MLflow UI for viewing experiments.

    Returns:
        str: Command to run in terminal.

    Example:
        >>> print(get_mlflow_ui_command())
        >>> # Output: mlflow ui --backend-store-uri file:./mlruns
    """
    return f"mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}"


def print_mlflow_info() -> None:
    """
    Print current MLflow configuration information.

    Useful for debugging and confirming setup.
    """
    print("=" * 70)
    print("MLflow Configuration")
    print("=" * 70)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment Name: {DEFAULT_EXPERIMENT_NAME}")
    print(f"Artifact Location: {DEFAULT_ARTIFACT_LOCATION}")
    print(f"Current Run ID: {get_current_run_id() or 'No active run'}")
    print(f"\nTo view experiments in browser, run:")
    print(f"  {get_mlflow_ui_command()}")
    print("=" * 70)


# =============================================================================
# MIGRATION GUIDE TO REMOTE MLFLOW SERVER
# =============================================================================
"""
TO MIGRATE TO REMOTE MLFLOW SERVER:
====================================

1. Deploy MLflow server (one-time setup):

   # On server machine:
   mlflow server \
       --backend-store-uri postgresql://user:pass@localhost/mlflow \
       --default-artifact-root s3://my-bucket/mlflow-artifacts \
       --host 0.0.0.0 \
       --port 5000

2. Update this file:

   MLFLOW_TRACKING_URI = "http://your-mlflow-server:5000"

3. All team members use the same configuration - no other code changes needed!

4. Benefits:
   - Centralized experiment tracking
   - Team can compare each other's runs
   - Better for production deployment
   - Can use cloud storage for artifacts (S3, GCS, etc.)

5. Authentication (if needed):

   import os
   os.environ['MLFLOW_TRACKING_USERNAME'] = 'your-username'
   os.environ['MLFLOW_TRACKING_PASSWORD'] = 'your-password'
"""
