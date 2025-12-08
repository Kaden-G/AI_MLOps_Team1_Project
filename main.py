from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import os
from pydantic import BaseModel, Field
import pandas as pd
from utils import BrainTumorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from utils import CNN, NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import AsyncGenerator
import torch.optim as optim
from enum import Enum
import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AvailableModels(str, Enum):
    CNN = "cnn"
    NN = "nn"


class TrainRequest(BaseModel):
    dataset_path: str = Field(description="The path to the dataset on the host.")
    test_size: float = Field(default=0.2, description="The percentage (as a decimal) of the data to use for a test dataset.")
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.001)
    momentum: float = Field(default=0.9)
    num_epochs: int = Field(default=10)
    save_path: str = Field(description="The path to save the model to on the host. Include the model name.")
    model_type: AvailableModels = Field(default=AvailableModels.CNN)
    confusion_matrix_save_path: str | None = None


app = FastAPI(
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
    optimizer = optim.SGD(net.parameters(), lr=request.learning_rate, momentum=request.momentum)


    for epoch in range(request.num_epochs):  # loop over the dataset multiple times
        yield f"EPOCH {epoch} ----------------------------\n"

        net.train()
        training_loss = 0.0
        true_positives = 0
        false_negatives = 0
        false_positives = 0
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

            true_positives += (outputs.squeeze().round() == labels.squeeze()).count_nonzero()
            false_negatives += torch.logical_and((outputs.squeeze().round() == 0.0), (labels.squeeze() == 1.0)).count_nonzero()
            false_positives += torch.logical_and((outputs.squeeze().round() == 1.0), (labels.squeeze() == 0.0)).count_nonzero()

            # print(f"OUTPUTS: {outputs.squeeze().round()}")
            # print(f"CASTED: {torch.logical_and((outputs.squeeze().round() == 0.0), (labels.squeeze() == 1.0)).count_nonzero()}")
            # print(f"LABELS: {labels.squeeze()}")

            # assert False
        yield f"TRAINING---\n"
        yield f"Average Training Loss for Epoch {epoch}: {training_loss/len(train_dataset)}\n"
        yield f"Training Accuracy for Epoch {epoch}: {true_positives/len(train_dataset)}\n"
        yield f"Training Recall for Epoch {epoch}: {true_positives/(true_positives + false_negatives)}\n"
        yield f"Training Precision for Epoch {epoch}: {true_positives/(true_positives + false_positives)}\n"


        # assert False

        net.eval()
        validation_loss = 0.0
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        true_y_labels = []
        predictions = []
        with torch.no_grad():
            for i, data in enumerate(validation_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                validation_loss += loss

                true_positives += (outputs.squeeze().round() == labels.squeeze()).count_nonzero()
                false_negatives += torch.logical_and((outputs.squeeze().round() == 0.0), (labels.squeeze() == 1.0)).count_nonzero()
                false_positives += torch.logical_and((outputs.squeeze().round() == 1.0), (labels.squeeze() == 0.0)).count_nonzero()

                true_y_labels.extend(labels.tolist())
                predictions.extend(outputs.round().tolist())

        yield f"VALIDATION---\n"
        yield f"Average Validation Loss for Epoch {epoch}: {validation_loss/len(validation_dataset)}\n"
        yield f"Validation Accuracy for Epoch {epoch}: {true_positives/len(validation_dataset)}\n"
        yield f"Validation Recall for Epoch {epoch}: {true_positives/(true_positives + false_negatives)}\n"
        yield f"Validation Precision for Epoch {epoch}: {true_positives/(true_positives + false_positives)}\n"


        # Saving at every epoch - we can change this if we want later.
        torch.save(net.state_dict(), request.save_path)
        yield f"Saving to: {request.save_path}\n"
    
    
        if epoch+1 >= request.num_epochs:
            cm = confusion_matrix(predictions, true_y_labels)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.savefig(request.confusion_matrix_save_path)



@app.post("/train")
async def train(request: TrainRequest):
    return StreamingResponse(training_generator(request))
    





