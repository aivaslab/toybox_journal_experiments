"""
Dataset module for balanced svhn
"""
import torchvision.datasets as datasets
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import numpy as np

import utils

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


class BalancedSVHN(torchdata.Dataset):
    """
    Balanced SVHN
    """
    def __init__(self, root, train=True, hypertune=True, transform=None):
        self.root = root
        self.train = train
        self.hypertune = hypertune
        self.transform = transform
        if self.hypertune or self.train:
            self.split = 'train'
        else:
            self.split = 'test'
        if self.train:
            self.indices_file = "../data/svhn_balanced_train_indices.npy"
        elif self.hypertune:
            self.indices_file = "../data/svhn_balanced_val_indices.npy"
        else:
            self.indices_file = "../data/svhn_balanced_test_indices.npy"
        print("Loading indices from {}".format(self.indices_file))
        super(BalancedSVHN, self).__init__()
        self.dataset = datasets.SVHN(root=self.root, split=self.split, transform=self.transform)
        self.indices = np.load(self.indices_file)
        
    def __getitem__(self, item):
        actual_item = self.indices[item]
        img, label = self.dataset[actual_item]
        return item, img, label
    
    def __len__(self):
        return len(self.indices)
    
    
if __name__ == "__main__":
    trnsfrm = transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD)])
    dataset = BalancedSVHN(root="../data/", train=True, hypertune=False, transform=trnsfrm)
    print(len(dataset))
    dataloader = torchdata.DataLoader(dataset, batch_size=64, shuffle=True)
    print(utils.online_mean_and_sd(dataloader))
    