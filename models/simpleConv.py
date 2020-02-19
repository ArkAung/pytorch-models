"""

Simple Convolutional Neural Network meant to work with 32x32x3 images

"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleConv(nn.Module):

    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    net = SimpleConv()

    # Create a random 3-channel input of dimension 32x32 and mini-batch size of 1.
    input_mat = torch.randn(1,3,32,32)
    
    # Forward pass
    output = net(input_mat)