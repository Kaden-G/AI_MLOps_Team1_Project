"""
Quick training script to create a test model
This trains for just 2 epochs to quickly generate a model file for testing
"""
import os
import pandas as pd
from utils import BrainTumorDataset, CNN
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Configuration
DATASET_PATH = "data/initial"
MODEL_SAVE_PATH = "models/brain_tumor_cnn_test.pt"
NUM_EPOCHS = 2  # Just 2 epochs for quick testing
BATCH_SIZE = 32
TEST_SIZE = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("Quick Model Training Script")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Model save path: {MODEL_SAVE_PATH}")
print()

# Load dataset
print("Loading dataset...")
file_names = os.listdir(f"{DATASET_PATH}/Brain Tumor/Brain Tumor")
file_paths = sorted(
    [f"{DATASET_PATH}/Brain Tumor/Brain Tumor/{x}" for x in file_names],
    key=lambda x: int(x.split(".jpg")[0].split("Image")[1])
)

labels = pd.read_csv(f"{DATASET_PATH}/Brain Tumor.csv")["Class"].tolist()
print(f"Total images: {len(file_paths)}")
print(f"Total labels: {len(labels)}")

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    file_paths, labels, test_size=TEST_SIZE, shuffle=True, random_state=42
)
print(f"Training images: {len(x_train)}")
print(f"Test images: {len(x_test)}")

# Create datasets
train_dataset = BrainTumorDataset(file_paths=x_train, labels=y_train)
validation_dataset = BrainTumorDataset(file_paths=x_test, labels=y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

# Create model
print("\nInitializing CNN model...")
model = CNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'=' * 60}")

    # Training phase
    model.train()
    training_loss = 0.0
    for i, (inputs, labels_batch) in enumerate(train_dataloader):
        inputs = inputs.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels_batch.squeeze())
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        if (i + 1) % 20 == 0:
            print(f"  Batch {i + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

    avg_train_loss = training_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    validation_loss = 0.0
    num_correct = 0
    with torch.no_grad():
        for inputs, labels_batch in validation_dataloader:
            inputs = inputs.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels_batch.squeeze())
            validation_loss += loss.item()

            num_correct += (outputs.squeeze().round() == labels_batch.squeeze()).count_nonzero()

    avg_val_loss = validation_loss / len(validation_dataloader)
    accuracy = num_correct / len(validation_dataset)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2%}")

# Save model
print(f"\n{'=' * 60}")
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved successfully!")
print(f"{'=' * 60}")
print("\nTraining complete! Model ready for testing with /predict endpoint.")
