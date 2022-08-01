import torch
import pickle
import csv
import cv2
import numpy as np
import logging
import torch.utils.data as torchdata


class DataLoaderGeneric(torchdata.Dataset):
    """
    This class
    """
    
    def __init__(self, root, train=True, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        self.train = train
        self.transform = transform
        self.root = root
        self.fraction = fraction
        self.hypertune = hypertune
        self.equal_div = equal_div
        self.logger = logging.getLogger(__name__)
        
        if self.train:
            if self.hypertune:
                self.images_file = self.root + "dev.pickle"
                self.labels_file = self.root + "dev.csv"
            else:
                self.images_file = self.root + "train.pickle"
                self.labels_file = self.root + "train.csv"
        else:
            if self.hypertune:
                self.images_file = self.root + "val.pickle"
                self.labels_file = self.root + "val.csv"
            else:
                self.images_file = self.root + "test.pickle"
                self.labels_file = self.root + "test.csv"
        
        self.images = pickle.load(open(self.images_file, "rb"))
        self.labels = list(csv.DictReader(open(self.labels_file, "r")))
        if self.train:
            if self.fraction < 1.0:
                len_all_images = len(self.images)
                rng = np.random.default_rng(0)
                if self.equal_div:
                    len_images_class = len_all_images // 12
                    len_train_images_class = int(self.fraction * len_images_class)
                    self.selected_indices = []
                    for i in range(12):
                        self.logger.debug(", ".join([str(i * len_images_class), str((i + 1) * len_images_class),
                                                     str(len(self.selected_indices))]))
                        all_indices = np.arange(i * len_images_class, (i + 1) * len_images_class)
                        sel_indices = rng.choice(all_indices, len_train_images_class, replace=False)
                        self.selected_indices = self.selected_indices + list(sel_indices)
                    self.logger.debug(len(self.selected_indices))
                else:
                    len_train_images = int(len_all_images * self.fraction)
                    self.selected_indices = rng.choice(len_all_images, len_train_images, replace=False)
            else:
                self.selected_indices = np.arange(len(self.images))
    
    def __len__(self):
        if self.train:
            return len(self.selected_indices)
        else:
            return len(self.images)
    
    def __getitem__(self, index):
        if self.train:
            item = self.selected_indices[index]
        else:
            item = index
        im = np.array(cv2.imdecode(self.images[item], 3))
        label = int(self.labels[item]["Class ID"])
        if self.transform is not None:
            im = self.transform(im)
        return index, im, label


if __name__ == "__main__":
    data = DataLoaderGeneric(root="../data_12/IN-12/", train=True, hypertune=True, fraction=0.01)
    print(len(data))
