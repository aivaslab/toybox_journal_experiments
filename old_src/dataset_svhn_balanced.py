"""
Dataset module for balanced svhn
"""
import torchvision.datasets as datasets
import torch.utils.data as torchdata
import numpy as np


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
    dataset = BalancedSVHN(root="../data/", train=True, hypertune=False)
    print(len(dataset))
    counts = [0] * 10
    for idx in range(len(dataset)):
        _, _, label = dataset[idx]
        counts[label] += 1
    print(counts)
