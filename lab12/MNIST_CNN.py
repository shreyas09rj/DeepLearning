#!/usr/bin/python

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


total_start_time = time.time()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = datasets.MNIST(
    root="/home/ibab/PycharmProjects/PyTorchProject/lab06/data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.MNIST(
    root="/home/ibab/PycharmProjects/PyTorchProject/lab06/data",
    train=False,
    download=True,
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


class BetterCNN(nn.Module):

    def __init__(self):
        super(BetterCNN, self).__init__()

        # ----------------------------------------------------
        # First convolution block
        # Input: 1 x 28 x 28
        # Output: 32 x 28 x 28
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)


        # ----------------------------------------------------
        # Second convolution block
        # Input: 32 x 14 x 14
        # Output: 64 x 14 x 14
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)


        # ----------------------------------------------------
        # Third convolution block
        # Input: 64 x 7 x 7
        # Output: 128 x 7 x 7
        # ----------------------------------------------------

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)


        # ----------------------------------------------------
        # Max Pooling
        # Reduces spatial dimensions by half
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(2, 2)


        # ----------------------------------------------------
        # Dropout
        # Used to reduce overfitting
        # ----------------------------------------------------

        self.dropout = nn.Dropout(0.5)


        # ----------------------------------------------------
        # Adaptive Average Pooling
        # Produces fixed 3 x 3 output
        # ----------------------------------------------------

        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))


        # ----------------------------------------------------
        # Fully connected layers
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            128 * 3 * 3,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)


        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)


        # ----------------------------------------------------
        # Block 3
        # ----------------------------------------------------

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)


        # ----------------------------------------------------
        # Adaptive pooling
        # ----------------------------------------------------

        x = self.adaptive_pool(x)


        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = torch.flatten(x, 1)


        # ----------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------

        x = F.relu(self.fc1(x))


        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        x = self.dropout(x)


        # ----------------------------------------------------
        # Final output
        # 10 classes: digits 0-9
        # ----------------------------------------------------

        x = self.fc2(x)

        return x


# ============================================================
# 7. CREATE MODEL
# ============================================================

model = BetterCNN().to(device)

print(model)

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


training_start_time = time.time()

num_epochs = 10

for epoch in range(num_epochs):

    # Training mode
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        # Move data to CPU/GPU
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


        # Get predicted class
        pred = output.argmax(dim=1)


        # Total number of images
        total += target.size(0)


        # Number of correct predictions
        correct += pred.eq(target).sum().item()

    scheduler.step()


    train_accuracy = 100.0 * correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Accuracy: {train_accuracy:.2f}%"
    )


training_end_time = time.time()

training_time = training_end_time - training_start_time

print(
    f"\nTraining time: {training_time:.2f} seconds"
)

print(
    f"Training time: {training_time / 60:.2f} minutes"
)



testing_start_time = time.time()

model.eval()

correct = 0
test_loss = 0.0


with torch.no_grad():

    for data, target in test_loader:

        # Move data to CPU/GPU
        data = data.to(device)
        target = target.to(device)


        # Forward pass
        output = model(data)


        # Calculate test loss
        test_loss += criterion(
            output,
            target
        ).item()


        # Get prediction
        pred = output.argmax(
            dim=1,
            keepdim=True
        )


        # Count correct predictions
        correct += pred.eq(
            target.view_as(pred)
        ).sum().item()


test_loss /= len(test_loader)

accuracy = 100.0 * correct / len(test_loader.dataset)


print(
    f"\nTest set: Average loss: {test_loss:.4f}, "
    f"Accuracy: {correct}/{len(test_loader.dataset)} "
    f"({accuracy:.2f}%)"
)



testing_end_time = time.time()

testing_time = testing_end_time - testing_start_time

print(
    f"Testing time: {testing_time:.2f} seconds"
)

print(
    f"Testing time: {testing_time / 60:.2f} minutes"
)



total_end_time = time.time()

total_time = total_end_time - total_start_time

print(
    f"\nTotal execution time: "
    f"{total_time:.2f} seconds"
)

print(
    f"Total execution time: "
    f"{total_time / 60:.2f} minutes"
)