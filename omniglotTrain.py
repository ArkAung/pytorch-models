from OmniglotDataset import OmniglotData
from models.SiameseNetwork import SiameseNetwork
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import torch

if __name__ == "__main__":
    transform_pipeline = transforms.Compose([transforms.Grayscale(),
                                            transforms.ToTensor()])
    train_data = OmniglotData(root='./data/Omniglot/images_background', transform=transform_pipeline)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    net = SiameseNetwork()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for e in range(4):
        running_loss = 0.0
        batch_size = 0
        for i, data in enumerate(train_loader):
            img1, img2, similar = data

            optimizer.zero_grad()
            output = net(img1, img2)
            loss = criterion(output, similar)
            loss.backward()
            optimizer.step()

            running_loss += loss
            batch_size += len(data[0])
        
        print("Epoch [{}/{}] -- Loss: {}".format(e,4, running_loss/batch_size))

    torch.save(net.state_dict(), "saved_model/simaese_trained.pth")
        
