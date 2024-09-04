import sys
import os

# Add the project directory to the system path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from models.teacher_model import TeacherModel
from data.cifar10 import get_cifar10_data
from utils.utils import set_device, save_model
from evaluation.evaluate import evaluate
import numpy as np

def train_teacher(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def main():
    # Set device
    device = set_device()
    print(f"Using device: {device}")

    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 30
    model_save_path = './models/Pretrained_teacher_model_new.pth'

    # Get CIFAR-10 data
    train_loader, test_loader = get_cifar10_data(batch_size)

    # Initialize the teacher model
    model = TeacherModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Variable to accumulate total accuracy and predictions
    total_test_accuracy = 0.0
    all_preds = []
    all_labels = []

    # Training loop with evaluation
    print("==> Starting training...")
    for epoch in range(num_epochs):
        # Train the teacher model
        train_loss = train_teacher(model, train_loader, criterion, optimizer, device)
        
        # Evaluate the model and gather predictions
        test_loss, test_accuracy, preds, labels = evaluate(model, test_loader, criterion, device, return_preds=True)
        
        # Accumulate test accuracy
        total_test_accuracy += test_accuracy
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        # Adjust the learning rate
        scheduler.step()
        
        # Print training and evaluation metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Calculate mean test accuracy
    mean_test_accuracy = total_test_accuracy / num_epochs
    print(f"Mean Test Accuracy over {num_epochs} epochs: {mean_test_accuracy:.2f}%")

    # Save the model
    save_model(model, model_save_path)

    # Save predictions and true labels for later analysis
    np.save('./visualizations/Pretrained_predictions1.npy', np.array(all_preds))
    np.save('./visualizations/Pretrained_labels1.npy', np.array(all_labels))

if __name__ == "__main__":
    main()
