import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

img = Image.open("frame.png")

transform = transforms.ToTensor()

tensor = transform(img).unsqueeze(0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))
    
class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(in_dim, out_dim),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
    def forward(self, x):
        return self.block(x)

class LamineYamal(nn.Module):
    def __init__(self, in_channels=3, in_width=800, in_height=600, n_actions=7):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 1024)
        self.conv2 = ConvBlock(1024, 256)
        self.conv3 = ConvBlock(256, 64)
        self.conv4 = ConvBlock(64, 16)
        self.conv5 = ConvBlock(16, 4)
        self.flatten = nn.Flatten()
        self.linear1 = LinearBlock(1472, 1024)
        self.linear2 = LinearBlock(1024, 256)
        self.linear3 = LinearBlock(256, 64)
        self.linear4 = LinearBlock(64, n_actions)

    def forward(self, x):
        out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        flat = self.flatten(out)
        print(flat.shape)
        final = self.linear4(self.linear3(self.linear2(self.linear1(flat))))
        return final