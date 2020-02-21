"""
    VAE for 784 flattened input (MNIST)
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400) 
        self.fc21 = nn.Linear(400, 40) # Mean
        self.fc22 = nn.Linear(400, 40) # Standard Deviation
        self.fc3 = nn.Linear(40, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE+KLD

def train(model, dataset_size, data_loader,
            optimizer, epoch):
    for i in range(epoch):
        running_loss = 0
        for batch_idx, data in enumerate(data_loader):
            image, label = data[0], data[1]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(image)
            loss = vae_loss(recon_x, image, mu, logvar)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        print("[{}/{} -- Loss: {}]".format(i,epoch,running_loss/dataset_size))

if __name__ == "__main__":
    net = VAE()
    optim = optim.Adam(net.parameters(), lr=0.001)
    train_dataset = MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    train(net, len(train_dataset), train_loader, optim, 2)    
