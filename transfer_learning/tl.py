#!/usr/bin/python

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler

from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import numpy as np
import os
import time

from PIL import Image


if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 60)
print("DEVICE:", device)
print("=" * 60)

data_transforms = {

    # TRAINING TRANSFORMS
    "train": transforms.Compose([

        # Random crop and resize to 224 x 224
        transforms.RandomResizedCrop(224),

        # Randomly flip image horizontally
        transforms.RandomHorizontalFlip(),

        # Convert image to tensor
        transforms.ToTensor(),

        # ImageNet normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),


    # VALIDATION TRANSFORMS

    "val": transforms.Compose([

        # Resize image
        transforms.Resize(256),

        # Center crop to 224 x 224
        transforms.CenterCrop(224),

        # Convert to tensor
        transforms.ToTensor(),

        # ImageNet normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}



data_dir = "data/hymenoptera_data"


image_datasets = {

    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        data_transforms[x]
    )

    for x in ["train", "val"]
}

dataloaders = {

    x: torch.utils.data.DataLoader(

        image_datasets[x],

        batch_size=4,

        shuffle=True,

        num_workers=2

    )

    for x in ["train", "val"]
}

dataset_sizes = {

    x: len(image_datasets[x])

    for x in ["train", "val"]

}

class_names = image_datasets["train"].classes


print("\nClasses:")
print(class_names)

print("\nDataset sizes:")
print(dataset_sizes)

def imshow(inp, title=None):

    # Convert tensor to NumPy
    inp = inp.numpy()

    inp = inp.transpose((1, 2, 0))


    # ImageNet mean
    mean = np.array(
        [0.485, 0.456, 0.406]
    )


    # ImageNet standard deviation
    std = np.array(
        [0.229, 0.224, 0.225]
    )

    inp = std * inp + mean


    # Keep values between 0 and 1
    inp = np.clip(
        inp,
        0,
        1
    )


    # Display image
    plt.imshow(inp)


    if title is not None:

        plt.title(title)


    plt.axis("off")

    plt.show()

inputs, classes = next(
    iter(dataloaders["train"])
)


# Make image grid
out = make_grid(inputs)


titles = [
    class_names[x]
    for x in classes
]


print("\nDisplaying training images...")


imshow(
    out,
    title=" | ".join(titles)
)


print("\nLoading pretrained ResNet-18...")


model = models.resnet18(
    weights="IMAGENET1K_V1"
)


print("\nOriginal final layer:")
print(model.fc)


# 12. GET NUMBER OF INPUT FEATURES


num_ftrs = model.fc.in_features


print(
    "\nNumber of features entering final layer:",
    num_ftrs
)

model.fc = nn.Linear(
    num_ftrs,
    2
)


print("\nNew final layer:")
print(model.fc)


model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(

    model.parameters(),

    lr=0.001,

    momentum=0.9
)


# 17. LEARNING RATE SCHEDULER

scheduler = lr_scheduler.StepLR(

    optimizer,

    # Every 7 epochs
    step_size=7,

    # Learning rate becomes:
    #
    # old LR x 0.1

    gamma=0.1
)


# 18. TRAINING FUNCTION

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25
):

    start_time = time.time()

    best_model_weights = None

    best_acc = 0.0

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []


    # EPOCH LOOP

    for epoch in range(num_epochs):


        print()
        print(
            f"Epoch {epoch + 1}/{num_epochs}"
        )

        print("-" * 50)


        # Reset values for this epoch

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        epoch_train_correct = 0
        epoch_val_correct = 0


        # TRAIN + VALIDATION

        for phase in ["train", "val"]:

            if phase == "train":

                model.train()


            # EVALUATION MODE
            else:

                model.eval()


            running_loss = 0.0

            running_corrects = 0


            # LOOP THROUGH BATCHES

            for inputs, labels in dataloaders[phase]:


                # Move images to GPU/CPU
                inputs = inputs.to(device)

                labels = labels.to(device)


                # CLEAR OLD GRADIENTS

                optimizer.zero_grad()


                # FORWARD PASS

                with torch.set_grad_enabled(
                    phase == "train"
                ):


                    # Neural network prediction
                    outputs = model(inputs)


                    # Get class with highest score
                    _, preds = torch.max(
                        outputs,
                        1
                    )


                    # Calculate loss
                    loss = criterion(
                        outputs,
                        labels
                    )


                    # BACKPROPAGATION

                    if phase == "train":

                        # Calculate gradients
                        loss.backward()


                        # Update weights
                        optimizer.step()


                # ACCUMULATE LOSS

                running_loss += (

                    loss.item()
                    * inputs.size(0)

                )


                # COUNT CORRECT PREDICTIONS

                running_corrects += (

                    torch.sum(
                        preds == labels
                    )

                )

            # CALCULATE EPOCH LOSS

            epoch_loss = (

                running_loss
                / dataset_sizes[phase]

            )



            # CALCULATE EPOCH ACCURACY
            epoch_acc = (

                running_corrects.double()
                / dataset_sizes[phase]

            )


            # PRINT RESULTS


            print(

                f"{phase.upper():5s} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {epoch_acc:.4f}"

            )

            # SAVE TRAINING HISTORY

            if phase == "train":

                epoch_train_loss = epoch_loss

                epoch_train_correct = epoch_acc


            else:

                epoch_val_loss = epoch_loss

                epoch_val_correct = epoch_acc

            # SAVE BEST MODEL

            if (

                phase == "val"

                and epoch_acc > best_acc

            ):

                best_acc = epoch_acc


                # Copy weights
                best_model_weights = {

                    key: value.cpu().clone()

                    for key, value in model.state_dict().items()}


        # UPDATE LEARNING RATE

        scheduler.step()


        # SAVE HISTORY
        train_losses.append(
            epoch_train_loss
        )

        val_losses.append(
            epoch_val_loss
        )

        train_accs.append(epoch_train_correct.item())

        val_accs.append(epoch_val_correct.item())

    # LOAD BEST MODEL

    model.load_state_dict(best_model_weights)


    model = model.to(device)


    # TRAINING TIME
    total_time = (time.time() - start_time)

    print()
    print("=" * 60)

    print(
        f"Training complete in "
        f"{total_time // 60:.0f}m "
        f"{total_time % 60:.0f}s"
    )

    print(
        f"Best validation accuracy: "
        f"{best_acc:.4f}"
    )

    print("=" * 60)


    # Return everything needed for visualization
    return (model,train_losses,val_losses,train_accs,val_accs)

# 19. TRAIN MODEL

model, train_losses, val_losses, train_accs, val_accs = (

    train_model(

        model,

        criterion,

        optimizer,

        scheduler,

        num_epochs=25

    )

)


# 20. VISUALIZE TRAINING AND VALIDATION LOSS

plt.figure(figsize=(8, 5))


plt.plot(
    train_losses,
    label="Training Loss"
)


plt.plot(
    val_losses,
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid()

plt.show()

plt.figure(figsize=(8, 5))


plt.plot(
    train_accs,
    label="Training Accuracy"
)


plt.plot(
    val_accs,
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid()

plt.show()


def visualize_predictions(
    model,
    num_images=6
):


    # Put model in evaluation mode

    model.eval()


    images_shown = 0


    plt.figure(
        figsize=(12, 8)
    )


    with torch.no_grad():


        # Get validation data

        for inputs, labels in dataloaders["val"]:


            # Move data to device

            inputs = inputs.to(device)

            labels = labels.to(device)


            # Prediction

            outputs = model(inputs)


            # Get predicted class

            _, preds = torch.max(
                outputs,
                1
            )

            for j in range(inputs.size(0)):

                if images_shown >= num_images:

                    break

                images_shown += 1

                # Create subplot

                ax = plt.subplot(
                    2,
                    3,
                    images_shown
                )

                # Remove axes

                ax.axis("off")


                # True class

                true_class = class_names[
                    labels[j].item()
                ]


                # Predicted class

                predicted_class = class_names[preds[j].item()]

                # Title

                ax.set_title(

                    f"True: {true_class}\n"
                    f"Predicted: {predicted_class}"

                )


                # Convert image for display

                image = inputs[
                    j
                ].cpu()
                image = image.numpy().transpose((1, 2, 0))

                # Undo normalization

                mean = np.array(
                    [0.485, 0.456, 0.406]
                )

                std = np.array(
                    [0.229, 0.224, 0.225]
                )


                image = (
                    image * std
                    + mean
                )

                image = np.clip(
                    image,
                    0,
                    1
                )


                # Display

                ax.imshow(image)


            if images_shown >= num_images:

                break


    plt.tight_layout()

    plt.show()

visualize_predictions(
    model,
    num_images=6
)


# 24. CUSTOM IMAGE PREDICTION

def predict_custom_image(
    model,
    image_path
):


    # LOAD IMAGE

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = data_transforms[
        "val"
    ](image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    # PREDICTION

    model.eval()


    with torch.no_grad():

        output = model(
            image_tensor
        )


        # Get highest score

        _, prediction = torch.max(
            output,
            1
        )


    # CLASS NAME

    predicted_class = class_names[
        prediction.item()
    ]


    print()
    print("=" * 60)

    print(
        "Image:",
        image_path
    )

    print(
        "Predicted class:",
        predicted_class
    )

    print("=" * 60)


    # DISPLAY IMAGE

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(image)

    plt.title(
        f"Predicted: {predicted_class}"
    )

    plt.axis("off")

    plt.show()

predict_custom_image(model,"myant.jpeg")

