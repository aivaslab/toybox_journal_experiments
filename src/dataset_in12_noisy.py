"""Implementation of noisy labels version of IN-12 dataset"""
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.utils.data as torchdata

import dataset_imagenet12
import utils

IN12_MEAN = (0.4541, 0.4845, 0.4980)
IN12_STD = (0.2928, 0.2738, 0.2756)

TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 0.1894, 0.1775)

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"


class DatasetIN12Noisy(dataset_imagenet12.DataLoaderGeneric):
    """
    This dataset implements the noisy labels version of IN12 dataset
    
    """
    def __init__(self, root, indices, train=True, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        super().__init__(root=root, train=train, transform=transform, fraction=fraction, hypertune=hypertune,
                         equal_div=equal_div)
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item):
        index = self.indices[item]
        
        if self.train:
            it = self.selected_indices[index]
        else:
            it = index
        im = np.array(cv2.imdecode(self.images[it], 3))
        label = int(self.labels[it]["Class ID"])
        if self.transform is not None:
            im = self.transform(im)
        return (item, it), im, label
        
        
if __name__ == "__main__":
    trnsfrm = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Resize(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
                                  ])
    num_images = 1000
    idxs = np.random.default_rng().choice(3600, 1000, replace=False)
    dataset = DatasetIN12Noisy(root=IN12_DATA_PATH, indices=idxs, train=True, transform=trnsfrm,
                               fraction=0.2, hypertune=True)
    print(len(dataset))
    dataloader = torchdata.DataLoader(dataset, batch_size=64, shuffle=True)
    print(utils.online_mean_and_sd(dataloader))
