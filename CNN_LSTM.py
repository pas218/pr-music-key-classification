import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=24, num_filters=128, num_hidden_units=256, dropout_rate=0):
        super(CNN_LSTM, self).__init__()
        
        # -------- Convolutional Blocks --------
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.pool2 = nn.MaxPool2d((4, 2), stride=(4, 2))

        self.conv3 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * num_filters)
        self.pool3 = nn.MaxPool2d((4, 2), stride=(4, 2))

        self.conv4 = nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2 * num_filters)
        self.pool4 = nn.MaxPool2d((4, 2), stride=(4, 2))

        # -------- LSTM --------
        # We will infer feature size dynamically
        self.lstm = nn.LSTM(
            input_size=2 * num_filters,  # will adjust after reshape
            hidden_size=num_hidden_units,
            batch_first=True
        )

        # ------- Dropout ----------
        self.dropout_conv = nn.Dropout2d(p=dropout_rate)
        self.dropout_lstm = nn.Dropout(p=dropout_rate)

        # -------- Fully Connected --------
        self.fc = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        
        # Conv Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout_conv(x)

        # x shape: (B, C, H, W)
        B, C, H, W = x.size()

        # -------- Prepare for LSTM --------
        # Treat width as time dimension
        # reshape to (B, W, C*H)
        x = x.permute(0, 3, 1, 2)   # (B, W, C, H)
        x = x.contiguous().view(B, W, C * H)

        # Adjust LSTM input size dynamically if needed
        if x.size(-1) != self.lstm.input_size:
            self.lstm = nn.LSTM(
                input_size=x.size(-1),
                hidden_size=self.lstm.hidden_size,
                batch_first=True
            ).to(x.device)

        # LSTM
        x, _ = self.lstm(x)

        # Take last time step (OutputMode="last")
        x = x[:, -1, :]

        x = self.dropout_lstm(x)

        # Fully connected
        x = self.fc(x)

        return x