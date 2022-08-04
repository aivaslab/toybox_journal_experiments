"""
Datasets for MNIST and SVHN which match dataset format for IN-12 and Toybox
"""
import torchvision.datasets as datasets
import torch.utils.data as torchdata


class DatasetMNIST(torchdata.Dataset):
    """
    Dataset class for MNIST
    """
    def __init__(self, root, train, transform):
        super(DatasetMNIST, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        
        self.dataset = datasets.MNIST(root=self.root, train=self.train, transform=self.transform)
        
    def __getitem__(self, item):
        img, label = self.dataset[item]
        return item, img, label
    
    def __len__(self):
        return len(self.dataset)
