
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import custom_dataset as cd
from cnn import SimpleCNN
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image

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


# 2. Define Transform pipeline
#transform = transforms.Compose([
#    transforms.Lambda(lambda x: x[:3, :, :]), 
#    transforms.Resize((450, 600), antialias=True),
#])

transform = v2.Compose([
    v2.Lambda(lambda x: x[:3, :, :]),
    v2.Resize((450, 600), antialias=True),
    v2.ToDtype(torch.float32, scale=True) # Recommended: converts to float and scales to [0, 1]
])

batch_size = 4
num_epochs = 1
soundfont = "arachnosf"

dataset = cd.CustomImageDataset(annotations_file=f'./dataset/{soundfont}/labels.csv', img_dir=f'./dataset/{soundfont}/images', transform=transform)

dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

#testset = cd.CustomImageDataset(annotations_file=f'./dataset/{soundfont}/labels.csv', img_dir=f'./dataset/{soundfont}/images', transform=transform)

#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


image, label = dataset.__getitem__(33)
#print(image.dtype)
#exit()
#to_pil = transforms.ToPILImage()
#img = to_pil(image)

# Open in the default system image viewer
#img.show()

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SimpleCNN(num_classes=24).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

counter = 0
# Training Loop
model.train() # Set model to training mode
for epoch in range(num_epochs):
    print(epoch)
    for images, labels in dataset_loader:
        print(counter)
        counter += 1
        #print("1")
        images, labels = images.to(device), labels.to(device)
        #print("2")
        # Forward pass
        outputs = model(images)
        #print("3")
        loss = criterion(outputs, labels)
        #print("4")
        # Backward pass and optimization
        optimizer.zero_grad() # Clear gradients from previous step
        #print("5")
        loss.backward()       # Compute gradients
        #print("6")
        optimizer.step()       # Update weights
        #print("7")


path = "./dataset/arachnosf/images/spect_005_arachnosf.png"
img = decode_image(path)
img_tensor = transform(img).unsqueeze(0) # Add batch dimension: [1, 3, 450, 600]

# 2. Perform Inference
model.eval() # Set to evaluation mode
with torch.no_grad(): # Disable gradient calculation for efficiency
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    
    # Get the predicted class index
    _, predicted_class = torch.max(output, 1)
    print(f"Predicted Class Index: {predicted_class.item()}")