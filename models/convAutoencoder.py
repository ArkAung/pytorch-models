"""
  Convolutional Autoencoders: Simple 2 conv layer and 2 transposed conv layer
  A good way to figure out the strides and kernel sizes to ensure that you are upsampling right 
  is to debug it with random image input and check the output sizes.
"""
import torch.nn as nn

class ConvAutoencoder(nn.Module):
  def __init__(self):
    super(ConvAutoencoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=(1,1)),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(4, 2, 3, padding=(1,1)),
        nn.ReLU(),
        nn.MaxPool2d((2,2))
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(2, 4, 2, stride=(2,2)),
        nn.ReLU(),
        nn.ConvTranspose2d(4, 1, 2, stride=(2,2)),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x