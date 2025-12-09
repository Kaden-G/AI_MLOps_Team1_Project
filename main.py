from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import os
from pydantic import BaseModel, Field
import pandas as pd
from utils import BrainTumorDataset, preprocess_image, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from utils import CNN, NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import AsyncGenerator, Optional, Dict, Any
import torch.optim as optim
from enum import Enum
from PIL import Image
import io
from model_loader import load_model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AvailableModels(str, Enum):
    CNN = "cnn"
    NN = "nn"


class TrainRequest(BaseModel):
    dataset_path: str = Field(
        description="The path to the dataset on the host.",
        example="/Users/username/data/initial"
    )
    test_size: float = Field(
        default=0.2,
        description="The percentage (as a decimal) of the data to use for a test dataset.",
        example=0.2
    )
    batch_size: int = Field(
        default=32,
        description="Number of samples per batch",
        example=64
    )
    num_epochs: int = Field(
        default=10,
        description="Number of training epochs",
        example=20
    )
    save_path: str = Field(
        description="The path to save the model to on the host. Include the model name.",
        example="/Users/username/models/production_model"
    )
    model_type: AvailableModels = Field(
        default=AvailableModels.CNN,
        description="Type of model architecture",
        example="cnn"
    )
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate for optimizer",
        example=0.001
    )
    momentum: float = Field(
        default=0.9,
        description="Momentum for SGD optimizer",
        example=0.9
    )
    early_stopping_patience: int = Field(
        default=5,
        description="Number of epochs to wait for improvement before stopping",
        example=5
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
        example=42
    )
    best_model_path: Optional[str] = Field(
        default=None,
        description="Path to save the best model. If None, appends '_best' to save_path",
        example="/Users/username/models/production_model_best"
    )


class PredictResponse(BaseModel):
    predicted_class: int = Field(
        description="Predicted class: 0 = No Tumor, 1 = Tumor",
        example=1
    )
    probability: float = Field(
        description="Probability of tumor presence (0.0 to 1.0)",
        example=0.8523
    )
    confidence_percentage: float = Field(
        description="Confidence as percentage",
        example=85.23
    )
    interpretation: str = Field(
        description="Human-readable interpretation of the prediction",
        example="TUMOR DETECTED - High concern (probability: 85.23%)"
    )


class TrainResponse(BaseModel):
    """Response model for training endpoint with comprehensive metrics"""
    status: str = Field(
        description="Training status",
        example="completed"
    )
    final_metrics: Dict[str, float] = Field(
        description="Final validation metrics",
        example={
            "val_accuracy": 0.737,
            "val_precision": 0.801,
            "val_recall": 0.902,
            "val_f1": 0.848,
            "val_loss": 0.523
        }
    )
    training_info: Dict[str, Any] = Field(
        description="Training configuration and results",
        example={
            "epochs_completed": 15,
            "early_stopped": True,
            "best_epoch": 10,
            "total_train_samples": 3009,
            "total_val_samples": 753
        }
    )
    model_paths: Dict[str, str] = Field(
        description="Paths to saved models",
        example={
            "best_model": "/Users/username/models/production_model_best",
            "final_model": "/Users/username/models/production_model"
        }
    )


app = FastAPI(
    title="Brain Tumor Classification API",
    description="""
    ## Production-Ready Brain Tumor Detection System

    Complete ML pipeline for training and predicting brain tumors from MRI images.

    ### Key Features:
    - **High Performance**: 90% recall for tumor detection (minimizes missed diagnoses)
    - **Direct Image Upload**: No file paths needed for predictions
    - **Cached Model Loading**: One-time model loading for fast inference
    - **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score
    - **Early Stopping**: Prevents overfitting with configurable patience
    - **Deterministic Training**: Reproducible results with fixed random seeds

    ### Endpoints:
    - **POST /train**: Train a new model with comprehensive metrics
    - **POST /predict**: Predict tumor presence from uploaded MRI image
    - **GET /health_check**: Check API health status

    ### Model Performance:
    - Validation Accuracy: 73.7%
    - Validation Recall: **90.2%** (critical for medical use)
    - Validation Precision: 80.1%
    - Validation F1 Score: ~84.8%

    ### Team:
    - Partner 1: Enhanced Training Pipeline
    - Partner 2: Deterministic Preprocessing
    - Partner 3: Production API (this implementation)
    """,
    version="2.0.0",
    contact={
        "name": "AI/MLOps Team 1",
        "email": "team1@jhu.edu"
    }
)

@app.get(
    "/",
    summary="Root Endpoint",
    description="Returns basic API information and status",
    tags=["Health"]
)
async def root():
    return {
        "message": "Brain Tumor Classification API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "train": "/train",
            "predict": "/predict",
            "health": "/health_check",
            "docs": "/docs"
        }
    }

@app.get(
    "/health_check",
    summary="Health Check",
    description="Check if the API is running and responding to requests",
    tags=["Health"]
)
async def health_check():
    return {
        "status": "ok",
        "api": "Brain Tumor Classification",
        "version": "2.0.0"
    }



async def train_model_structured(request: TrainRequest) -> TrainResponse:
    """
    Train model and return structured JSON response with all metrics.

    This replaces the streaming generator for production use.
    Returns comprehensive training results as structured JSON.
    """
    try:
        # Set random seed for reproducibility (Partner 2 requirement)
        set_seed(request.random_seed)

        # Load dataset
        file_names = os.listdir(request.dataset_path.rstrip("/") + "/Brain Tumor/Brain Tumor")
        file_paths = sorted([request.dataset_path.rstrip("/") + f"/Brain Tumor/Brain Tumor/{x}" for x in file_names], key=lambda x: int(x.split(".jpg")[0].split("Image")[1]))

        labels = pd.read_csv(request.dataset_path.rstrip("/") + "/Brain Tumor.csv")["Class"].tolist()

        # Split with deterministic seed (Partner 2 requirement)
        x_train, x_test, y_train, y_test = train_test_split(
            file_paths, labels,
            test_size=request.test_size,
            shuffle=True,
            random_state=request.random_seed
        )

        train_dataset = BrainTumorDataset(file_paths=x_train, labels=y_train, normalize=True)
        validation_dataset = BrainTumorDataset(file_paths=x_test, labels=y_test, normalize=True)

        train_dataloader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=request.batch_size)

        # Initialize model
        match request.model_type:
            case AvailableModels.CNN:
                net = CNN().to(DEVICE)
            case AvailableModels.NN:
                net = NN().to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(net.parameters(), lr=request.learning_rate, momentum=request.momentum)

        # Early stopping setup (Partner 1 requirement)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = request.best_model_path if request.best_model_path else f"{request.save_path}_best"
        best_epoch = 0
        early_stopped = False

        # Track final metrics
        final_metrics = {}
        epoch_history = []

        # Training loop
        for epoch in range(request.num_epochs):
            # Training phase
            net.train()
            training_loss = 0.0
            train_predictions = []
            train_targets = []

            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

                # Collect predictions for metrics
                preds = outputs.squeeze().round().cpu().detach().numpy()
                targs = labels.squeeze().cpu().numpy()
                train_predictions.extend(preds.flatten() if preds.ndim > 0 else [preds.item()])
                train_targets.extend(targs.flatten() if targs.ndim > 0 else [targs.item()])

            # Calculate training metrics
            avg_train_loss = training_loss / len(train_dataset)
            train_accuracy = (torch.tensor(train_predictions) == torch.tensor(train_targets)).float().mean().item()
            train_precision = precision_score(train_targets, train_predictions, zero_division=0)
            train_recall = recall_score(train_targets, train_predictions, zero_division=0)
            train_f1 = f1_score(train_targets, train_predictions, zero_division=0)

            # Validation phase
            net.eval()
            validation_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for i, data in enumerate(validation_dataloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = net(inputs)
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    validation_loss += loss.item()

                    preds = outputs.squeeze().round().cpu().numpy()
                    targs = labels.squeeze().cpu().numpy()
                    val_predictions.extend(preds.flatten() if preds.ndim > 0 else [preds.item()])
                    val_targets.extend(targs.flatten() if targs.ndim > 0 else [targs.item()])

            # Calculate validation metrics
            avg_val_loss = validation_loss / len(validation_dataset)
            val_accuracy = (torch.tensor(val_predictions) == torch.tensor(val_targets)).float().mean().item()
            val_precision = precision_score(val_targets, val_predictions, zero_division=0)
            val_recall = recall_score(val_targets, val_predictions, zero_division=0)
            val_f1 = f1_score(val_targets, val_predictions, zero_division=0)

            # Store epoch metrics
            epoch_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_recall": val_recall
            })

            # Update final metrics (will be overwritten each epoch)
            final_metrics = {
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_loss": avg_val_loss
            }

            # Save current model every epoch
            torch.save(net.state_dict(), request.save_path)

            # Early stopping and best model checkpointing (Partner 1 requirement)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_epoch = epoch
                torch.save(net.state_dict(), best_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= request.early_stopping_patience:
                    early_stopped = True
                    break

        # Calculate confusion matrix for final epoch
        cm = confusion_matrix(val_targets, val_predictions)

        # Prepare response
        return TrainResponse(
            status="completed",
            final_metrics=final_metrics,
            training_info={
                "epochs_completed": epoch + 1,
                "early_stopped": early_stopped,
                "best_epoch": best_epoch,
                "total_train_samples": len(train_dataset),
                "total_val_samples": len(validation_dataset),
                "learning_rate": request.learning_rate,
                "batch_size": request.batch_size,
                "confusion_matrix": cm.tolist()
            },
            model_paths={
                "best_model": best_model_path,
                "final_model": request.save_path
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


async def training_generator(request: TrainRequest) -> AsyncGenerator:
    # Set random seed for reproducibility (Partner 2 requirement)
    set_seed(request.random_seed)
    yield f"Set random seed to {request.random_seed} for reproducibility\n"

    file_names = os.listdir(request.dataset_path.rstrip("/") + "/Brain Tumor/Brain Tumor")
    file_paths = sorted([request.dataset_path.rstrip("/") + f"/Brain Tumor/Brain Tumor/{x}" for x in file_names], key=lambda x: int(x.split(".jpg")[0].split("Image")[1]))

    labels = pd.read_csv(request.dataset_path.rstrip("/") + "/Brain Tumor.csv")["Class"].tolist()

    # Split with deterministic seed (Partner 2 requirement)
    x_train, x_test, y_train, y_test = train_test_split(
        file_paths, labels,
        test_size=request.test_size,
        shuffle=True,
        random_state=request.random_seed
    )

    train_dataset = BrainTumorDataset(file_paths=x_train, labels=y_train, normalize=True)
    validation_dataset = BrainTumorDataset(file_paths=x_test, labels=y_test, normalize=True)

    train_dataloader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=request.batch_size)

    match request.model_type:
        case AvailableModels.CNN:
            net = CNN().to(DEVICE)
        case AvailableModels.NN:
            net = NN().to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=request.learning_rate, momentum=request.momentum)

    # Early stopping setup (Partner 1 requirement)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = request.best_model_path if request.best_model_path else f"{request.save_path}_best"

    yield f"Training starting with {len(train_dataset)} training samples, {len(validation_dataset)} validation samples\n"
    yield f"Learning rate: {request.learning_rate}, Momentum: {request.momentum}\n"
    yield f"Early stopping patience: {request.early_stopping_patience} epochs\n"

    for epoch in range(request.num_epochs):
        yield f"\nEPOCH {epoch} ----------------------------\n"

        # Training phase
        net.train()
        training_loss = 0.0
        train_predictions = []
        train_targets = []

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # Collect predictions for metrics
            preds = outputs.squeeze().round().cpu().detach().numpy()
            targs = labels.squeeze().cpu().numpy()
            # Handle both single samples and batches
            train_predictions.extend(preds.flatten() if preds.ndim > 0 else [preds.item()])
            train_targets.extend(targs.flatten() if targs.ndim > 0 else [targs.item()])

        # Calculate training metrics (Partner 1 requirement)
        avg_train_loss = training_loss / len(train_dataset)
        train_accuracy = (torch.tensor(train_predictions) == torch.tensor(train_targets)).float().mean().item()
        train_precision = precision_score(train_targets, train_predictions, zero_division=0)
        train_recall = recall_score(train_targets, train_predictions, zero_division=0)
        train_f1 = f1_score(train_targets, train_predictions, zero_division=0)

        yield f"Training Metrics - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}\n"

        # Validation phase
        net.eval()
        validation_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for i, data in enumerate(validation_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                validation_loss += loss.item()

                # Collect predictions for metrics
                preds = outputs.squeeze().round().cpu().numpy()
                targs = labels.squeeze().cpu().numpy()
                # Handle both single samples and batches
                val_predictions.extend(preds.flatten() if preds.ndim > 0 else [preds.item()])
                val_targets.extend(targs.flatten() if targs.ndim > 0 else [targs.item()])

        # Calculate validation metrics (Partner 1 requirement - comprehensive metrics)
        avg_val_loss = validation_loss / len(validation_dataset)
        val_accuracy = (torch.tensor(val_predictions) == torch.tensor(val_targets)).float().mean().item()
        val_precision = precision_score(val_targets, val_predictions, zero_division=0)
        val_recall = recall_score(val_targets, val_predictions, zero_division=0)
        val_f1 = f1_score(val_targets, val_predictions, zero_division=0)

        yield f"Validation Metrics - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}\n"

        # Save current model every epoch
        torch.save(net.state_dict(), request.save_path)
        yield f"Saved current model to: {request.save_path}\n"

        # Early stopping and best model checkpointing (Partner 1 requirement)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(net.state_dict(), best_model_path)
            yield f"✓ New best model! Saved to: {best_model_path} (Val Loss: {best_val_loss:.4f})\n"
        else:
            epochs_without_improvement += 1
            yield f"No improvement for {epochs_without_improvement} epoch(s) (Best Val Loss: {best_val_loss:.4f})\n"

            if epochs_without_improvement >= request.early_stopping_patience:
                yield f"\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {request.early_stopping_patience} epochs)\n"
                yield f"Best validation loss: {best_val_loss:.4f}\n"
                yield f"Best model saved at: {best_model_path}\n"
                break

    yield f"\nTraining complete! Final best validation loss: {best_val_loss:.4f}\n"
    yield f"Best model available at: {best_model_path}\n"
    yield f"Latest model available at: {request.save_path}\n"



@app.post(
    "/train",
    response_model=TrainResponse,
    summary="Train Brain Tumor Classification Model",
    description="""
    Train a CNN or NN model for brain tumor detection with comprehensive metrics.

    **Features:**
    - **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1 score
    - **Early Stopping**: Configurable patience to prevent overfitting
    - **Best Model Checkpointing**: Saves best performing model separately
    - **Deterministic Training**: Reproducible results with fixed random seed
    - **Structured Output**: Returns JSON with all metrics and model paths

    **Training Process:**
    1. Load and split dataset (configurable test_size)
    2. Initialize model architecture (CNN or NN)
    3. Train with SGD optimizer
    4. Validate after each epoch
    5. Save best model based on validation loss
    6. Early stop if no improvement

    **What You Get:**
    - Final validation metrics (accuracy, precision, recall, F1)
    - Training configuration details
    - Paths to saved models (best + final)
    - Early stopping information

    **Typical Training Time:**
    - 2 epochs: ~2-3 minutes (demo)
    - 20 epochs: ~10-15 minutes (production)
    """,
    tags=["Training"]
)
async def train(request: TrainRequest):
    return await train_model_structured(request)


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Brain Tumor from MRI Image",
    description="""
    Upload a brain MRI image to predict whether it contains a tumor.

    **Features:**
    - Direct image upload (no file path needed)
    - One-time model loading (cached for performance)
    - Returns probability, confidence, and interpretation
    - Uses same preprocessing as training pipeline

    **Medical Context:**
    - Class 0: No Tumor Detected
    - Class 1: Tumor Detected
    - Model achieves 90% recall (minimizes missed tumors)

    **How to Use:**
    1. Select your trained model file path
    2. Choose model type (cnn or nn)
    3. Upload your brain MRI image
    4. Receive instant prediction with interpretation
    """
)
async def predict(
    file: UploadFile = File(..., description="Brain MRI image file (JPG, PNG, etc.)"),
    model_path: str = Form(..., description="Path to trained model weights (.pt or .pth)"),
    model_type: str = Form(default="cnn", description="Model architecture type (cnn or nn)")
):
    """
    Partner 3's Enhanced Prediction Endpoint

    Accepts direct image upload and uses cached model loading for performance.
    """
    try:
        # Validate model file exists
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

        # Validate model type
        if model_type not in ["cnn", "nn"]:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}. Must be 'cnn' or 'nn'")

        # Read uploaded image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Load model using cached loader (ONE-TIME LOADING)
        model = load_model(model_path, model_type)

        # Preprocess the image using Partner 2's function (supports PIL Image)
        image_tensor = preprocess_image(image, normalize=True)
        image_tensor = image_tensor.to(DEVICE)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.squeeze().item()  # Get scalar value

        # Determine predicted class (threshold = 0.5)
        predicted_class = 1 if probability >= 0.5 else 0

        # Calculate confidence percentage
        if predicted_class == 1:
            confidence = probability * 100
        else:
            confidence = (1 - probability) * 100

        # Generate human-readable interpretation
        if predicted_class == 1:
            interpretation = f"TUMOR DETECTED - High concern (probability: {probability:.2%})"
        else:
            interpretation = f"NO TUMOR DETECTED - Low concern (probability: {probability:.2%})"

        return PredictResponse(
            predicted_class=predicted_class,
            probability=probability,
            confidence_percentage=round(confidence, 2),
            interpretation=interpretation
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



