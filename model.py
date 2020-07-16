import torch
import torch.nn as nn
import torch.nn.functional as F


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
