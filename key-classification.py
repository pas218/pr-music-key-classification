
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import custom_dataset as cd
from cnn import SimpleCNN
from CNN_LSTM import CNN_LSTM
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm

def get_first_3_channels_lambda(x):
    return x[:3, :, :]

labels_map = {
    0: 'Ab:maj',
    1: 'A:maj',
    2: 'Bb:maj',
    3: 'B:maj',
    4: 'C:maj',
    5: 'Db:maj',
    6: 'D:maj',
    7: 'Eb:maj',    
    8: 'E:maj',
    9: 'F:maj',
    10: 'Gb:maj',
    11: 'G:maj',
    12: 'Ab:min',
    13: 'A:min',
    14: 'Bb:min',
    15: 'B:min',
    16: 'C:min',
    17: 'Db:min',
    18: 'D:min',
    19: 'Eb:min',
    20: 'E:min',
    21: 'F:min',
    22: 'Gb:min',
    23: 'G:min'
}

soundfont_map = {
    0: "arachnosf",
    1: "fzero",
    2: "genuser",
    3: "pokemonredgreen",
    4: "sonic2piano"
}

TRAIN_SET_PROPORTION = .8
VAL_SET_PROPORTION = 1 - TRAIN_SET_PROPORTION


# 2. Define Transform pipeline
#transform = transforms.Compose([
#    transforms.Lambda(lambda x: x[:3, :, :]), 
#    transforms.Resize((450, 600), antialias=True),
#])

transform = v2.Compose([
    v2.Lambda(get_first_3_channels_lambda),
    v2.Grayscale(num_output_channels=1),
    v2.Resize((450, 600), antialias=True),
    v2.ToDtype(torch.float32, scale=True) # Recommended: converts to float and scales to [0, 1]
])


def main():
    batch_size = 25
    num_epochs = 5

    dataset = cd.CustomImageDataset(annotations_file=f'./dataset/labels.csv', img_dir=f'./dataset/spects', num_soundfonts=5, soundfont_map=soundfont_map, transform=transform)
    train_size = int(TRAIN_SET_PROPORTION * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    #testset = cd.CustomImageDataset(annotations_file=f'./dataset/{soundfont}/labels.csv', img_dir=f'./dataset/{soundfont}/images', transform=transform)

    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


    # image, label = dataset.__getitem__(33)
    #print(image.dtype)
    #exit()
    #to_pil = transforms.ToPILImage()
    #img = to_pil(image)

    # Open in the default system image viewer
    #img.show()

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNN_LSTM(num_classes=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, leave=True)
        for batch_idx, (images, labels) in enumerate(train_loop):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad() # Clear gradients from previous step
            loss.backward()       # Compute gradients
            optimizer.step()       # Update weights

            train_loop.set_description(f"Training Epoch [{epoch + 1}/{num_epochs}]")


    # 2. Perform Inference

    total_loss = 0
    correct = 0
    total = 0

    model.eval() # Set to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for efficiency
        val_loop = tqdm(val_loader, leave=True)
        for batch_idx, (images, labels) in enumerate(val_loop):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted_class = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted_class == labels).sum().item()
            val_loop.set_description(f"Validating Model...")
        
        # Get the predicted class index
    
    avg_val_loss = total_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct / total

    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")


if __name__ == '__main__':
    main()