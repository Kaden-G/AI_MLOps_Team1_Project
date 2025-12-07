from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import os
from pydantic import BaseModel, Field
import pandas as pd
from utils import BrainTumorDataset, preprocess_image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import CNN, NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import AsyncGenerator
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
    file_names = os.listdir(request.dataset_path.rstrip("/") + "/Brain Tumor/Brain Tumor")
    file_paths = sorted([request.dataset_path.rstrip("/") + f"/Brain Tumor/Brain Tumor/{x}" for x in file_names], key=lambda x: int(x.split(".jpg")[0].split("Image")[1]))

    labels = pd.read_csv(request.dataset_path.rstrip("/") + "/Brain Tumor.csv")["Class"].tolist()
    # File paths and labels are in order, split into train & test

    x_train, x_test, y_train, y_test = train_test_split(file_paths, labels, test_size=request.test_size, shuffle=True)

    train_dataset = BrainTumorDataset(file_paths=x_train, labels=y_train)
    validation_dataset = BrainTumorDataset(file_paths=x_test, labels=y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=request.batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=request.batch_size)

    match request.model_type:
        case AvailableModels.CNN:
            net = CNN().to(DEVICE)
        case AvailableModels.NN:
            net = NN().to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(request.num_epochs):  # loop over the dataset multiple times
        yield f"EPOCH {epoch} ----------------------------\n"

        net.train()
        training_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()

        yield f"Average Training Loss for Epoch {epoch}: {training_loss/len(train_dataset)}\n"

        net.eval()
        validation_loss = 0.0
        num_correct = 0
        with torch.no_grad():
            for i, data in enumerate(validation_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                validation_loss += loss

                num_correct += (outputs.squeeze().round() == labels.squeeze()).count_nonzero()

        yield f"Average Validation Loss for Epoch {epoch}: {validation_loss/len(validation_dataset)}\n"
        yield f"Validation Accuracy for Epoch {epoch}: {num_correct/len(validation_dataset)}\n"


        # Saving at every epoch - we can change this if we want later.
        torch.save(net.state_dict(), request.save_path)
        yield f"Saving to: {request.save_path}\n"



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



