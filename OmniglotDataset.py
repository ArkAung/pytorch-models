from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from models.SiameseNetwork import SiameseNetwork
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np


class OmniglotData(ImageFolder):
    """Omnitglot Dataset to load pairs of similar images and different images"""

    def __init__(self, root, transform=None, is_valid_file=None):
        super(OmniglotData, self).__init__(root, transform=transform, target_transform=None, is_valid_file=is_valid_file)
        self.samples_per_class = self._get_samples_per_class(self.classes)

    def _get_samples_per_class(self, class_names):
        dict_class_samples = {}
        for c in class_names:
            class_path = os.path.join(self.root, c)
            class_str_to_indx = self.class_to_idx[c] # Convert folder names to classes indexes
            file_paths = [os.path.join(class_path, x) for x in os.listdir(class_path)] # Get full file paths for a class c
            dict_class_samples[class_str_to_indx] = file_paths
        return dict_class_samples
   
    def __getitem__(self, index):
        """
            Get two samples from the same class if the index is odd number.
            Get two samples from different class if the index is even number.
        """
        path1, target = self.samples[index]
        if index % 2 == 0:
            samples_of_target = self.samples_per_class[target]
            
            is_similar = True
        else:
            other_class = target
            while (other_class == target):
                other_class = self.class_to_idx[np.random.choice(self.classes)]
            samples_of_target = self.samples_per_class[other_class]
            is_similar = False

        another_alphabet_root = np.random.choice(samples_of_target) 
        path2 = os.path.join(another_alphabet_root, np.random.choice(os.listdir(another_alphabet_root)))

        sample1 = self.loader(path1)
        sample2 = self.loader(path2)
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        
        return sample1, sample2, is_similar


train_dataset = OmniglotData(root='./data/Omniglot/images_background')
even = train_dataset[0]
odd = train_dataset[1]
train_dataloader = DataLoader(dataset=train_dataset, batch_size=100)

classes = train_dataset.classes

net = SiameseNetwork()