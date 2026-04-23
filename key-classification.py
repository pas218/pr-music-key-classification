
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
from key_classfn_utilities import *


TRAIN_SET_PROPORTION = .8
VAL_SET_PROPORTION = 1 - TRAIN_SET_PROPORTION
NUM_SOUNDFONTS = len(soundfont_map)

transform = v2.Compose([
    v2.Lambda(get_first_3_channels_lambda),
    v2.Grayscale(num_output_channels=1),
    v2.Resize((450, 600), antialias=True),
    v2.ToDtype(torch.float32, scale=True) # Recommended: converts to float and scales to [0, 1]
])


def main(dropout_rate):
    batch_size = 16
    num_epochs = 20
    num_filters = 128
    num_hidden_units = 256

    dataset = cd.CustomImageDataset(annotations_file=f'./dataset/labels.csv', img_dir=f'./dataset/spects', num_soundfonts=NUM_SOUNDFONTS, soundfont_map=soundfont_map, transform=transform)

    train_dataset, val_dataset = datasetsplit_coupled(dataset, int(len(dataset)/NUM_SOUNDFONTS), NUM_SOUNDFONTS, [TRAIN_SET_PROPORTION, VAL_SET_PROPORTION])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(num_classes=24, num_filters=num_filters, num_hidden_units=num_hidden_units, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.00075)

    # Training Loop
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, leave=True, disable=False)
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
        val_loop = tqdm(val_loader, leave=True, disable=True)
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

    torch.save(model, f'model_{int(val_accuracy)}acc.pt')

    return val_accuracy


if __name__ == '__main__':

    Dropouts = [.2, .3, .5]

    for dropout in Dropouts:
        curr_accuracy = main(dropout_rate=dropout)
        print(f"Model Acc: {curr_accuracy}, Dropout Rate: {dropout}")
            

    
