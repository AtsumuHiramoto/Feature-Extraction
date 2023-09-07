import torch
from torch.utils.data import DataLoader
# import torchvision
# from torchvision import transforms
# from torchvision.datasets import MNIST, CIFAR10  # CIFAR10もインポートしておく
import numpy as np
import matplotlib.pyplot as plt

class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 64)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 512)
        self.fc3 = torch.nn.Linear(512, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self, channel_patch=[3,8,16], channel_hand=[16,32,64]):
        super().__init__()
        org_size = 1152
        self.enc = Encoder(org_size)
        self.dec = Decoder(org_size)
    def forward(self, x, coordinates_cog=None):
        feature = self.enc(x)  # エンコード
        x = self.dec(feature)  # デコード
        return feature, x