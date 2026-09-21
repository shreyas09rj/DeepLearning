#!/usr/bin/python

# Cifar - 10

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from jinja2.compiler import F
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device :",device)

trnasform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    download=True,
    train=True,
    transform=trnasform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    download=True,
    train=False,
    transform=trnasform
)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=2)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=64,
                         shuffle=False,
                         num_workers=2)
classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")

images,labels = next(iter(train_loader))
image = images[0]
image = image /2 + 0.5
image = image.permute(1, 2, 0)
plt.imshow(image)
plt.title(classes[labels[0]])
plt.axis('off')
plt.show()

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CIFAR_CNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    total = 0
    correct = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct +=(predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy =  ( correct / total ) * 100
    epoch_error = 100 - epoch_accuracy
    train_losses.append(epoch_loss)
    test_losses.append(epoch_error)

    print(f"Epoch [{epoch+1}/{epochs}] ",
          f"Loss: {epoch_loss:.4f} ",
          f"Accuracy: {epoch_accuracy:.2f}%",
          f"Error: {epoch_error:.2f}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = ( correct / total ) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

plt.figure()
plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker='o'
)
plt.xlabel("Epochs")
plt.ylabel("Training loss")
plt.title("Training Loss")
plt.show()

plt.figure()
plt.plot(
    range(1, epochs + 1),
    test_losses,
    marker='o'
)
plt.xlabel("Epochs")
plt.ylabel("Testing loss")
plt.title("Testing Loss")
plt.show()
