import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(SimpleCNN, self).__init__()
        # Input: 3 x 450 x 600
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # 16 x 450 x 600
        self.pool = nn.MaxPool2d(2, 2) # Reduces size by half
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 32 x 225 x 300
        
        # Flatten size: 32 * (450/4) * (600/4) = 32 * 112 * 150 = 537600
        self.fc1 = nn.Linear(32 * 112 * 150, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is assumed to be (Batch, 3, 450, 600), normalized to [0,1]
        x = self.pool(F.relu(self.conv1(x))) # 16 x 225 x 300
        x = self.pool(F.relu(self.conv2(x))) # 32 x 112 x 150
        x = x.view(-1, 32 * 112 * 150) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x