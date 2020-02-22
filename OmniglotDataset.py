from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from models.SiameseNetwork import SiameseNetwork
import torch.optim as optim
import torch.nn as nn
import os



class OmniglotData(ImageFolder):
    """Omnitglot Dataset to load pairs of similar images and different images"""

    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None):
        super(OmniglotData, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        self.samples_per_class = self._get_samples_per_class(self.classes)

    def _get_samples_per_class(self, class_names):
        dict_class_samples = {}
        for c in class_names:
            class_path = os.path.join(self.root, c)
            class_str_to_indx = self.class_to_idx[c]
            dict_class_samples[class_str_to_indx] = os.listdir(class_path)
        return dict_class_samples
   
    def __getitem__(self, index):
        path, target = self.samples[index]

        self.class_to_idx
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


train_dataset = OmniglotData(root='./data/Omniglot/images_background')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=100)

classes = train_dataset.classes

net = SiameseNetwork()