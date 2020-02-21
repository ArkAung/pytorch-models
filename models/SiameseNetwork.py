"""
    Siamese network
    Follows the architecture from https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    Expects 105x105x1 input image
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(1, 64, 10),
                            nn.ReLU(),
                            nn.MaxPool2d((2,2)),
                            nn.Conv2d(64,128,7),
                            nn.ReLU(),
                            nn.MaxPool2d((2,2)),
                            nn.Conv2d(128,128,4),
                            nn.ReLU(),
                            nn.MaxPool2d((2,2)),
                            nn.Conv2d(128,256,4),
                            nn.ReLU()
                            )
        self.linear = nn.Sequential(
                        nn.Linear(9216,4096),
                        nn.Sigmoid()
                        )
        self.out = nn.Sequential(
                    nn.Linear(4096,1),
                    nn.Sigmoid()
                    )

    # Forward pass for one input
    def forward_brach(self, x):
        x = self.conv_block(x)
        x = x.view(-1)
        x = self.linear(x)
        return x

    # Forward pass with two inputs
    def forward(self, x1, x2):
        out1 = self.forward_brach(x1)
        out2 = self.forward_brach(x2)
        # Compute L1 component-wise distance
        distance = self.cal_distance(out1, out2)
        sig_out = self.out(distance)
        return sig_out

    def cal_distance(self, x1, x2):
        return torch.abs(x1 - x2)

if __name__=="__main__":
    # Test the network with random inputs
    img1 = torch.randn(1,1,105,105)
    img2 = torch.randn(1,1,105,105)
    net = SiameseNetwork()
    output = net(img1, img2)
    print(output)