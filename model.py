import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetComplete(nn.Module):

    def __init__(self):
        super(LeNetComplete, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Define forward pass of CNN

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = self.block1(x)

        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        x = x.view(-1, 4*4*16)

        # Apply first fully-connected block to input tensor
        x = self.block3(x)

        return F.log_softmax(x, dim=1)

class ClientNN(nn.Module):

    def __init__(self):

        super(ClientNN, self).__init__()

        #CNN
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):

        x = self.firstBlock(x)

        return x

class ServerNN(nn.Module):

    def __init__(self):

        super(ServerNN, self).__init__()

        #CNN
        self.secondBlock = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.thirdBlock = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):

        x = self.secondBlock(x)
        x = x.view(-1, 4*4*16)
        x = self.thirdBlock(x)

        return x
