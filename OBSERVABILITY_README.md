# Brain Tumor Classification - Observability Stack

**Johns Hopkins 635.603 - AI/ML Ops (Fall 2025)**
**Complete, Architecture-Agnostic MLOps Observability Layer**

---

## 🎯 Overview

This is a **complete, production-ready observability stack** for monitoring machine learning models in your brain tumor classification project. Everything is **model-agnostic** - it works with ANY CNN architecture your team chooses (basic CNN, ResNet, EfficientNet, custom models, etc.).

### Key Features

✅ **MLflow Experiment Tracking** - Reproducible ML experiments
✅ **Training Monitoring** - Epoch-by-epoch metric logging
✅ **Clinical-Grade Metrics** - Precision, recall, F1, ROC-AUC with medical interpretation
✅ **Data Quality Checks** - Image corruption and brightness monitoring
✅ **Drift Detection** - Catches when data or predictions change over time
✅ **Latency Monitoring** - Production performance tracking
✅ **Automated Alerting** - Threshold-based alerts for critical issues
✅ **Complete Documentation** - Heavy comments, docstrings, examples

---

## 📁 Project Structure

```
project_root/
├── observability/
│   ├── __init__.py              # Package initialization
│   ├── mlflow_config.py         # MLflow setup and configuration
│   ├── logger.py                # Training and inference logging
│   ├── metrics.py               # Evaluation metrics computation
│   ├── data_quality.py          # Image quality monitoring
│   ├── drift.py                 # Prediction and embedding drift detection
│   ├── alerts.py                # Automated alerting system
│   └── monitoring.py            # End-to-end monitoring utilities
├── notebooks/
│   └── observability_demo.ipynb # Complete demonstration notebook
└── OBSERVABILITY_README.md      # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn mlflow pillow jupyter
```

### 2. Run the Demo Notebook

```bash
cd notebooks
jupyter notebook observability_demo.ipynb
```

The notebook demonstrates all capabilities with simulated data. Perfect for:
- Understanding how the system works
- Presenting to your team/professor
- Testing before integrating with your model

### 3. View MLflow UI

After running the notebook:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open `http://localhost:5000` in your browser to explore logged experiments.

---

## 📚 Module Descriptions

### `mlflow_config.py` - Experiment Tracking

**Purpose:** Centralized MLflow configuration and experiment management.

**Key Functions:**
- `setup_mlflow()` - Initialize tracking (local or remote)
- `start_run()` - Begin logging experiment
- `end_run()` - Finalize experiment

**Why It Matters:**
- Reproducible experiments
- Compare different models/hyperparameters
- Team collaboration (when using remote server)

**Migration to Remote:**
```python
# Change this line in mlflow_config.py:
MLFLOW_TRACKING_URI = "http://your-mlflow-server:5000"
# All other code stays the same!
```

---

### `logger.py` - Training & Inference Logging

**Purpose:** Convenient logging functions for model lifecycle events.

**Key Functions:**
- `log_training_params()` - Log hyperparameters
- `log_epoch_metrics()` - Log per-epoch metrics
- `log_training_summary()` - Log final training stats
- `log_inference_batch()` - Log prediction results

**Architecture-Agnostic:**
```python
# Works with ANY model!
logger.log_training_params(
    learning_rate=0.001,
    architecture="your_model_name"  # "resnet50", "custom_cnn", etc.
)
```

---

### `metrics.py` - Evaluation Metrics

**Purpose:** Comprehensive, clinical-grade model evaluation.

**Key Functions:**
- `compute_classification_metrics()` - All metrics in one call
- `print_metrics()` - Formatted output with interpretation
- `find_optimal_threshold()` - Tune decision threshold

**Clinical Focus:**
- **Recall** prioritized (missing tumors = dangerous!)
- **Precision** important (false positives = unnecessary procedures)
- Both should be ≥ 80% per project goals

**Example:**
```python
from observability import metrics

results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)
metrics.print_metrics(results)

# Output includes:
# - Accuracy, Precision, Recall, F1, ROC-AUC
# - Confusion matrix
# - Clinical interpretation
# - Project goal assessment
```

---

### `data_quality.py` - Image Quality Monitoring

**Purpose:** Detect corrupted or unusual images before they reach the model.

**Key Functions:**
- `check_image_brightness()` - Calculate brightness
- `detect_corrupted_image()` - Check for corruption
- `analyze_dataset_quality()` - Full dataset analysis

**Production Use:**
```python
from observability import data_quality

# Check single image
is_bad, reason = data_quality.detect_corrupted_image("scan.jpg")
if is_bad:
    print(f"Rejected: {reason}")

# Analyze entire dataset
report = data_quality.analyze_dataset_quality(image_paths)
data_quality.print_quality_report(report)
```

---

### `drift.py` - Distribution Shift Detection

**Purpose:** Detect when model behavior or input data changes over time.

**Key Functions:**
- `detect_prediction_drift()` - Check if predictions changed
- `detect_embedding_drift()` - Check if inputs changed (advanced)
- `monitor_drift()` - Complete drift monitoring

**Why Drift Matters:**
- New scanner → different image characteristics
- Different patient population → different tumor rates
- Model degradation → predictions become unreliable

**Example:**
```python
from observability import drift

# Monitor weekly
drift_report = drift.monitor_drift(
    reference_predictions=baseline_preds,
    current_predictions=this_week_preds
)

drift.print_drift_report(drift_report)

if drift_report['overall_drift_detected']:
    # Alert team, investigate, consider retraining
```

**Optional Embedding Drift:**
If you extract embeddings from your model (e.g., second-to-last layer output), you can detect drift more sensitively:

```python
# Extract embeddings (model-specific)
embedding_model = create_embedding_extractor(your_model)
ref_embeddings = embedding_model.predict(train_data)
cur_embeddings = embedding_model.predict(production_data)

# Detect drift
drift_report = drift.monitor_drift(
    reference_predictions=ref_preds,
    current_predictions=cur_preds,
    reference_embeddings=ref_embeddings,
    current_embeddings=cur_embeddings
)
```

---

### `alerts.py` - Automated Alerting

**Purpose:** Threshold-based alerts for critical issues.

**Alert Types:**
1. **Performance** - Recall/precision below threshold
2. **Drift** - Prediction distribution changed
3. **Quality** - High corruption rate, unusual brightness
4. **Operational** - High latency, failures

**Configurable Thresholds:**
```python
from observability import alerts

# Use default thresholds (80% recall/precision)
alert_list = alerts.check_performance_alerts(metrics)

# Or customize
config = alerts.AlertConfig()
config.min_recall = 0.90  # Require 90% recall
config.min_precision = 0.85

alert_list = alerts.check_performance_alerts(metrics, config=config)
```

**Severities:**
- `critical` - Immediate action required (e.g., recall < 70%)
- `high` - Urgent attention needed
- `warning` - Monitor closely
- `info` - FYI

---

### `monitoring.py` - End-to-End Monitoring

**Purpose:** High-level utilities tying everything together.

**Key Functions:**
- `evaluate_batch()` - Complete evaluation pipeline
- `simulate_latency_samples()` - Generate test data
- `run_monitoring_demo()` - Full demonstration

**Example Production Pipeline:**
```python
from observability import monitoring, alerts

# Evaluate weekly batch
report = monitoring.evaluate_batch(
    y_true=week_labels,
    y_pred=week_predictions,
    y_prob=week_probabilities,
    batch_name=f"week_{week_num}",
    reference_predictions=baseline_preds
)

# Print summary
print(report['summary'])

# Handle alerts
if report['alerts']:
    for alert in report['alerts']:
        if alert.severity in ['critical', 'high']:
            send_urgent_notification(alert)
```

---

## 🎓 MLO Objectives Satisfied

| MLO | Requirement | How We Address It |
|-----|-------------|-------------------|
| **8.1** | Clear ML pipeline | Modular components with defined interfaces |
| **8.4** | Tracking/metadata | MLflow experiment tracking |
| **8.5** | Production-grade code | Error handling, logging, documentation |
| **8.6** | Performance evaluation | Comprehensive metrics with interpretation |
| **8.7** | Deployment monitoring | Drift, quality, latency monitoring |
| **14.1** | Final demonstration | Complete notebook and visualizations |

---

## 🏥 Clinical Safety Features

This observability stack prioritizes **patient safety**:

1. **Recall Monitoring** - Alerts if model misses tumors (< 80%)
2. **Precision Tracking** - Monitors false positive rate
3. **Drift Detection** - Catches degradation before harm occurs
4. **Data Quality** - Rejects corrupted/invalid images
5. **Latency Monitoring** - Ensures timely diagnoses

---

## 🔧 Integration with Your Model

The observability stack is **architecture-agnostic**. Here's how to integrate:

### During Training

```python
from observability import mlflow_config, logger

# Setup MLflow
mlflow_config.setup_mlflow()
mlflow_config.start_run(run_name="my_training_run")

# Log hyperparameters
logger.log_training_params(
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    optimizer="adam",
    loss_function="binary_crossentropy",
    architecture="your_model_name"  # Could be "resnet50", "custom_cnn", etc.
)

# Training loop
for epoch in range(epochs):
    # ... your training code ...

    # Log metrics
    logger.log_epoch_metrics(
        epoch=epoch,
        train_loss=train_loss,
        train_accuracy=train_acc,
        val_loss=val_loss,
        val_accuracy=val_acc,
        val_precision=val_prec,
        val_recall=val_rec,
        val_f1=val_f1
    )

# Log summary
logger.log_training_summary(
    total_epochs_completed=epochs,
    best_val_accuracy=best_acc,
    best_val_recall=best_recall,
    best_val_precision=best_prec,
    final_train_loss=final_loss,
    training_time_seconds=training_time
)

mlflow_config.end_run()
```

### During Evaluation

```python
from observability import metrics, alerts

# Get predictions from your model
y_pred = (model.predict(test_images) > 0.5).astype(int)
y_prob = model.predict(test_images)

# Compute metrics
results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)
metrics.print_metrics(results)

# Check alerts
alert_list = alerts.check_performance_alerts(results)
if alert_list:
    print("ALERTS DETECTED:")
    for alert in alert_list:
        print(f"  {alert}")
```

### In Production

```python
from observability import monitoring, drift, alerts

# Monitor weekly batches
weekly_report = monitoring.evaluate_batch(
    y_true=week_labels,
    y_pred=week_predictions,
    y_prob=week_probabilities,
    batch_name=f"production_week_{week}",
    reference_predictions=baseline_predictions
)

# Check drift
drift_report = drift.monitor_drift(
    reference_predictions=baseline_predictions,
    current_predictions=week_predictions
)

# Handle alerts
all_alerts = alerts.check_all_alerts(
    metrics=weekly_report['metrics'],
    drift_report=drift_report
)

if all_alerts:
    # Send notifications, log to dashboard, etc.
    pass
```

---

## 📊 Visualization Examples

The notebook generates these visualizations:

1. **training_history.png** - Loss and metric curves over epochs
2. **data_quality_brightness.png** - Image brightness distribution
3. **confusion_matrix.png** - TP, TN, FP, FN breakdown
4. **roc_curve.png** - ROC-AUC visualization
5. **drift_detection.png** - Prediction distribution comparison
6. **latency_monitoring.png** - Inference latency analysis
7. **alert_summary.png** - Alert dashboard

All plots are publication-quality, suitable for presentations and reports.

---

## 🎯 SMART Goals Enforcement

Your project commits to ≥80% precision and recall. The observability stack enforces this:

```python
# Automatic goal checking
results = metrics.compute_classification_metrics(y_true, y_pred, y_prob)

if results['recall'] < 0.80:
    # CRITICAL alert triggered
    # Alert message: "Recall below threshold! Missing tumors is clinically dangerous!"

if results['precision'] < 0.80:
    # HIGH/WARNING alert triggered
    # Alert message: "Precision below threshold. Too many false positives."
```

---

## 🚢 Production Deployment

### Local to Remote MLflow Migration

**Step 1:** Deploy MLflow server (one-time)

```bash
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --default-artifact-root s3://bucket/artifacts \
    --host 0.0.0.0 \
    --port 5000
```

**Step 2:** Update config (one line change)

```python
# In mlflow_config.py:
MLFLOW_TRACKING_URI = "http://your-server:5000"
```

**Step 3:** All code continues to work!

### Alert Notifications

Extend `alerts.py` to send notifications:

```python
def send_alert_notification(alert):
    if alert.severity in ['critical', 'high']:
        # Send email
        send_email(to="team@example.com", subject=f"ALERT: {alert.message}")

        # Send Slack message
        send_slack(channel="#ml-alerts", message=str(alert))

        # Page on-call engineer for critical
        if alert.severity == 'critical':
            pagerduty.trigger(alert)
```

### Continuous Monitoring

Set up a cron job or scheduled task:

```bash
# Run weekly monitoring
0 0 * * 0 python scripts/weekly_monitoring.py
```

```python
# scripts/weekly_monitoring.py
from observability import monitoring, alerts
import datetime

week = datetime.datetime.now().isocalendar()[1]

# Load data
y_true, y_pred, y_prob = load_weekly_predictions(week)
baseline = load_baseline_predictions()

# Monitor
report = monitoring.evaluate_batch(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    batch_name=f"week_{week}",
    reference_predictions=baseline
)

# Handle alerts
if report['alerts']:
    for alert in report['alerts']:
        send_notification(alert)
```

---

## 📖 Documentation Standards

Every module includes:

✅ **Module-level docstring** - Purpose and role
✅ **Function docstrings** - Parameters, returns, examples
✅ **Inline comments** - Explaining *why*, not just *what*
✅ **Usage examples** - At end of each module
✅ **Type hints** - For clarity (where helpful)

Example function documentation:

```python
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
        y_prob (array-like, optional): Prediction probabilities.
            Required for ROC-AUC calculation.
        positive_label (int): Which label is "positive" (default: 1 = tumor).

    Returns:
        dict: Dictionary containing:
            - 'accuracy': Overall accuracy
            - 'precision': Precision for tumor class
            - 'recall': Recall (sensitivity) - MOST CRITICAL!
            - 'f1': F1 score
            - 'roc_auc': Area under ROC curve (if y_prob provided)
            - ... and more

    Example:
        >>> results = compute_classification_metrics(y_true, y_pred, y_prob)
        >>> print(f"Recall: {results['recall']:.2%}")

    Note:
        This function is completely model-agnostic!
    """
```

---

## ❓ FAQ

### Q: Does this work with my model architecture?

**A:** Yes! The observability stack is **completely architecture-agnostic**. It works with:
- Basic CNNs
- Transfer learning (ResNet, EfficientNet, VGG, etc.)
- Custom architectures
- Any future model your team develops

It only needs standard outputs: predictions (y_pred), labels (y_true), and optionally probabilities (y_prob).

---

### Q: What if I don't have embeddings for drift detection?

**A:** No problem! Prediction drift (based on positive prediction rates) works without embeddings and is still very useful. Embedding drift is an **optional enhancement** you can add later if needed.

---

### Q: How do I customize alert thresholds?

**A:** Easy! Create a custom `AlertConfig`:

```python
from observability import alerts

config = alerts.AlertConfig()
config.min_recall = 0.90        # Require 90% instead of 80%
config.min_precision = 0.85
config.max_latency_ms = 500     # Stricter latency

alert_list = alerts.check_performance_alerts(metrics, config=config)
```

---

### Q: Can I use this without MLflow?

**A:** Yes, but MLflow is highly recommended for reproducibility and team collaboration. If you skip MLflow, you can still use:
- `metrics.py` for evaluation
- `data_quality.py` for quality checks
- `drift.py` for drift detection
- `alerts.py` for alerting

Just skip the `logger.py` functions that require an active MLflow run.

---

### Q: How do I demo this for the final presentation?

**A:** Run the `observability_demo.ipynb` notebook! It:
- Demonstrates all capabilities
- Generates visualizations
- Shows MLO objective alignment
- Takes ~5 minutes to execute
- Produces publication-quality plots

Perfect for live demo or pre-recorded presentation.

---

## 🎓 For Course Submission

When submitting or presenting this work:

### What to Highlight

1. **Architecture-Agnostic Design**
   - "Works with ANY model our team chooses"
   - "No hard-coded assumptions"

2. **MLO Objectives**
   - Show mapping to each MLO (8.1, 8.4, 8.5, 8.6, 8.7, 14.1)
   - Demonstrate how each module supports objectives

3. **Clinical Awareness**
   - "Recall prioritized for patient safety"
   - "Automated alerts prevent dangerous scenarios"
   - "Drift detection maintains model reliability"

4. **Production-Ready**
   - "Comprehensive error handling"
   - "Extensive documentation"
   - "Easy migration to team deployment"

### Suggested Demo Flow

1. **Introduction** (2 min)
   - Problem: Need observability for ML model
   - Solution: Complete, modular stack
   - Key feature: Architecture-agnostic

2. **Training Monitoring** (3 min)
   - Show MLflow setup
   - Run training simulation
   - Display learning curves

3. **Evaluation** (3 min)
   - Compute metrics
   - Show confusion matrix
   - Explain recall importance

4. **Production Monitoring** (4 min)
   - Data quality checks
   - Drift detection
   - Latency monitoring
   - Alerting

5. **Conclusion** (2 min)
   - MLO objectives satisfied
   - Ready for team integration
   - Questions

---

## 📞 Support & Questions

This observability stack is designed to be self-explanatory, but if you have questions:

1. **Check the code comments** - Every module has extensive inline documentation
2. **Read the module docstrings** - Each file has a header explaining its purpose
3. **Run the demo notebook** - Interactive examples of all features
4. **Review function docstrings** - Every function has usage examples

---

## 🙏 Acknowledgments

**Course:** Johns Hopkins 635.603 - AI/ML Ops (Fall 2025)
**Project:** Brain MRI Tumor Classification using CNNs
**Dataset:** Brain MRI Images for Brain Tumor Detection (Kaggle)

**Technologies:**
- MLflow (experiment tracking)
- scikit-learn (metrics)
- SciPy (statistical tests)
- NumPy/Pandas (data manipulation)
- Matplotlib/Seaborn (visualization)
- Pillow (image processing)

---

## ✅ Summary

You now have a **complete, production-ready observability stack** that:

✅ Works with ANY CNN architecture
✅ Satisfies ALL relevant MLO objectives
✅ Enforces project SMART goals (≥80% precision & recall)
✅ Provides clinical-grade monitoring
✅ Is fully documented and tested
✅ Ready for final presentation

**Next Steps:**
1. Run the demo notebook
2. Integrate with your model training code
3. Customize thresholds if needed
4. Present to team/professor
5. Deploy to production

**Good luck with your project! 🎓🏥**
