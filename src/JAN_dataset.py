"""Dataset implementations for JAN model"""
import csv
import cv2
import numpy as np
import pickle
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import utils


OFFICE31_AMAZON_MEAN = (0.7841, 0.7862, 0.7923)
OFFICE31_AMAZON_STD = (0.3201, 0.3182, 0.3157)
OFFICE31_DSLR_MEAN = (0.4064, 0.4487, 0.4709)
OFFICE31_DSLR_STD = (0.2025, 0.1949, 0.2067)
OFFICE31_WEBCAM_MEAN = (0.6172, 0.6187, 0.6120)
OFFICE31_WEBCAM_STD = (0.2589, 0.2568, 0.2519)


class Office31Amazon(torchdata.Dataset):
    """
    Dataset Class for Office-31 images for Amazon domain
    """
    IMAGES_PATH = "../../DATASETS/office-31/amazon.pkl"
    LABELS_PATH = "../../DATASETS/office-31/amazon.csv"
    
    def __init__(self, transform=None):
        self.transform = transform
        
        images_file = open(self.IMAGES_PATH, "rb")
        labels_file = open(self.LABELS_PATH, "r")
        self.images = pickle.load(images_file)
        self.labels = list(csv.DictReader(labels_file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        img, label = np.array(cv2.imdecode(self.images[item], 3)), int(self.labels[item]['Class ID'])
        if self.transform:
            img = self.transform(img)
        return item, img, label


class Office31DSLR(torchdata.Dataset):
    """
    Dataset Class for Office-31 images for DSLR domain
    """
    IMAGES_PATH = "../../DATASETS/office-31/dslr.pkl"
    LABELS_PATH = "../../DATASETS/office-31/dslr.csv"
    
    def __init__(self, transform=None):
        self.transform = transform
        
        images_file = open(self.IMAGES_PATH, "rb")
        labels_file = open(self.LABELS_PATH, "r")
        self.images = pickle.load(images_file)
        self.labels = list(csv.DictReader(labels_file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        img, label = np.array(cv2.imdecode(self.images[item], 3)), int(self.labels[item]['Class ID'])
        if self.transform:
            img = self.transform(img)
        return item, img, label


class Office31Webcam(torchdata.Dataset):
    """
    Dataset Class for Office-31 images for Webcam domain
    """
    IMAGES_PATH = "../../DATASETS/office-31/webcam.pkl"
    LABELS_PATH = "../../DATASETS/office-31/webcam.csv"
    
    def __init__(self, transform=None):
        self.transform = transform
        
        images_file = open(self.IMAGES_PATH, "rb")
        labels_file = open(self.LABELS_PATH, "r")
        self.images = pickle.load(images_file)
        self.labels = list(csv.DictReader(labels_file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        img, label = np.array(cv2.imdecode(self.images[item], 3)), int(self.labels[item]['Class ID'])
        if self.transform:
            img = self.transform(img)
        return item, img, label


if __name__ == "__main__":
    trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                  transforms.Resize(224),
                                  transforms.Normalize(mean=OFFICE31_AMAZON_MEAN, std=OFFICE31_AMAZON_STD)])
    dataset = Office31Amazon(transform=trnsfrm)
    print(len(dataset))
    dataloader = torchdata.DataLoader(dataset, batch_size=64, shuffle=False)
    print(utils.online_mean_and_sd(dataloader))

    trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                  transforms.Resize(224),
                                  transforms.Normalize(mean=OFFICE31_DSLR_MEAN, std=OFFICE31_DSLR_STD)])
    dataset = Office31DSLR(transform=trnsfrm)
    print(len(dataset))
    dataloader = torchdata.DataLoader(dataset, batch_size=64, shuffle=False)
    print(utils.online_mean_and_sd(dataloader))

    trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                  transforms.Resize(224),
                                  transforms.Normalize(mean=OFFICE31_WEBCAM_MEAN, std=OFFICE31_WEBCAM_STD)])
    dataset = Office31Webcam(transform=trnsfrm)
    print(len(dataset))
    dataloader = torchdata.DataLoader(dataset, batch_size=64, shuffle=False)
    print(utils.online_mean_and_sd(dataloader))
    