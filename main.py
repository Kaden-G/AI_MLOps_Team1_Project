from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import os
from pydantic import BaseModel, Field
import pandas as pd
from utils import BrainTumorDataset, preprocess_image, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from utils import CNN, NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import AsyncGenerator, Optional
import torch.optim as optim
from enum import Enum


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AvailableModels(str, Enum):
    CNN = "cnn"
    NN = "nn"


class TrainRequest(BaseModel):
    dataset_path: str = Field(description="The path to the dataset on the host.")
    test_size: float = Field(default=0.2, description="The percentage (as a decimal) of the data to use for a test dataset.")
    batch_size: int = Field(default=32)
    num_epochs: int = Field(default=10)
    save_path: str = Field(description="The path to save the model to on the host. Include the model name.")
    model_type: AvailableModels = Field(default=AvailableModels.CNN)
    learning_rate: float = Field(default=0.001, description="Learning rate for optimizer")
    momentum: float = Field(default=0.9, description="Momentum for SGD optimizer")
    early_stopping_patience: int = Field(default=5, description="Number of epochs to wait for improvement before stopping")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    best_model_path: Optional[str] = Field(default=None, description="Path to save the best model. If None, appends '_best' to save_path")


class PredictRequest(BaseModel):
    image_path: str = Field(description="Path to the brain MRI image to classify")
    model_path: str = Field(description="Path to the trained model file (.pt or .pth)")
    model_type: AvailableModels = Field(default=AvailableModels.CNN, description="Type of model architecture to load")


class PredictResponse(BaseModel):
    predicted_class: int = Field(description="Predicted class: 0 = No Tumor, 1 = Tumor")
    probability: float = Field(description="Probability of tumor presence (0.0 to 1.0)")
    confidence_percentage: float = Field(description="Confidence as percentage")
    interpretation: str = Field(description="Human-readable interpretation of the prediction")


app = FastAPI(
    title="Brain Tumor Classification API",
    description="API for training and predicting brain tumors from MRI images",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Test"}

@app.get("/health_check")
async def health_check():
    return {"status": "ok"}



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



@app.post("/train")
async def train(request: TrainRequest):
    return StreamingResponse(training_generator(request))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Partner 3's Prediction Endpoint

    Loads a trained model and predicts whether a brain MRI image contains a tumor.

    Returns:
        - predicted_class: 0 (No Tumor) or 1 (Tumor)
        - probability: Raw probability from model (0.0 to 1.0)
        - confidence_percentage: Confidence as percentage
        - interpretation: Human-readable result
    """
    try:
        # Validate image file exists
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_path}")

        # Validate model file exists
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")

        # Load the appropriate model architecture (from Partner 1)
        match request.model_type:
            case AvailableModels.CNN:
                model = CNN().to(DEVICE)
            case AvailableModels.NN:
                model = NN().to(DEVICE)

        # Load the trained weights
        model.load_state_dict(torch.load(request.model_path, map_location=DEVICE))
        model.eval()  # Set to evaluation mode

        # Preprocess the image using Partner 2's function
        image_tensor = preprocess_image(request.image_path)
        image_tensor = image_tensor.to(DEVICE)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.squeeze().item()  # Get scalar value

        # Determine predicted class (threshold = 0.5)
        predicted_class = 1 if probability >= 0.5 else 0

        # Calculate confidence percentage
        # For binary classification: confidence is distance from 0.5
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



