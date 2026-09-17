#!/usr/bin/python

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = (torch.accelerator.current_accelerator().type
          if torch.accelerator.is_available() else "cpu"
)

print("Using device:", device)

# load fashion MNIST
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# Training
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Test dataset
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Create Dataloader
batch_size = 64

train_dataloader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

# check batch shape
for x, y in test_dataloader:
    print("Shape of x", x.shape)
    print("Shape of y", y.shape)
    break


# Feedforward Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        #28*28 image into 784 features
        x = self.flatten(x)

        #forward propagation
        logits = self.network(x)
        return logits


model = NeuralNetwork().to(device)

print("Model :", model)

# loss function and optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)

            print(
                f"loss:{loss_value:>7f}"
                f"[{current:>5d}/{size:>5d}]"
            )


# testing
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            test_loss += loss_fn(pred, y).item()

            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches
    accuracy = correct / size

    print(
        f"test error:\n"
        f"accuracy:{accuracy:.4f}\n"
        f"avg loss:{test_loss:>8f}"
    )


epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")

    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer
    )

    test(
        test_dataloader,
        model,
        loss_fn
    )


torch.save(model.state_dict(), "model.pth")

loaded_model = NeuralNetwork().to(device)

loaded_model.load_state_dict(
    torch.load("model.pth", weights_only=True)
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

loaded_model.eval()

# Take first test image
x, y = test_data[0]

with torch.no_grad():
    x = x.to(device)

    pred = loaded_model(x)

    predicted_class = classes[pred.argmax(1).item()]
    actual_class = classes[y]

print("\nPrediction:")
print("Predicted :", predicted_class)
print("Actual    :", actual_class)





