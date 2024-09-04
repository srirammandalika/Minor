import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

import torch
import torch.optim as optim
import torch.nn as nn
from models.teacher_model import TeacherModel
from data.cifar10 import get_cifar10_data
from utils.utils import save_model

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 30
model_save_path = './models/teacher_model_Dis.pth'

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Get CIFAR-10 data
train_loader, test_loader = get_cifar10_data(batch_size)

# Initialize model, loss function, and optimizer
model = TeacherModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
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

    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the trained model
save_model(model, model_save_path)
print("Model training completed and weights saved.")
