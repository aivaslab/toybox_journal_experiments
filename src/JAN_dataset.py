"""Dataset implementations for JAN model"""
import csv
import cv2
import numpy as np
import pickle

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import utils
import dataset_mnist50
import dataset_svhn_balanced


MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)

SVHN_MEAN = (0.4377, 0.4437, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

OFFICE31_AMAZON_MEAN = (0.7841, 0.7862, 0.7923)
OFFICE31_AMAZON_STD = (0.3201, 0.3182, 0.3157)
OFFICE31_DSLR_MEAN = (0.4064, 0.4487, 0.4709)
OFFICE31_DSLR_STD = (0.2025, 0.1949, 0.2067)
OFFICE31_WEBCAM_MEAN = (0.6172, 0.6187, 0.6120)
OFFICE31_WEBCAM_STD = (0.2589, 0.2568, 0.2519)

IN12_MEAN = (0.4980, 0.4845, 0.4541)
IN12_STD = (0.2756, 0.2738, 0.2928)

TOYBOX_MEAN = (0.5199, 0.4374, 0.3499)
TOYBOX_STD = (0.1775, 0.1894, 0.1623)

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"
DATASETS = ['amazon', 'dslr', 'webcam', 'toybox', 'in12', "mnist50", "svhn-b"]


class Office31(torchdata.Dataset):
    """
    Dataset Class for Office-31 images
    """
    AMAZON_IMAGES_PATH = "../../DATASETS/office-31/amazon.pkl"
    AMAZON_LABELS_PATH = "../../DATASETS/office-31/amazon.csv"
    DSLR_IMAGES_PATH = "../../DATASETS/office-31/dslr.pkl"
    DSLR_LABELS_PATH = "../../DATASETS/office-31/dslr.csv"
    WEBCAM_IMAGES_PATH = "../../DATASETS/office-31/webcam.pkl"
    WEBCAM_LABELS_PATH = "../../DATASETS/office-31/webcam.csv"
    DOMAINS = ['amazon', 'dslr', 'webcam']
    
    def __init__(self, domain, transform=None):
        assert domain in self.DOMAINS
        self.domain = domain
        self.transform = transform
        
        if self.domain == 'amazon':
            self.IMAGES_PATH = self.AMAZON_IMAGES_PATH
            self.LABELS_PATH = self.AMAZON_LABELS_PATH
        elif self.domain == 'dslr':
            self.IMAGES_PATH = self.DSLR_IMAGES_PATH
            self.LABELS_PATH = self.DSLR_LABELS_PATH
        else:
            self.IMAGES_PATH = self.WEBCAM_IMAGES_PATH
            self.LABELS_PATH = self.WEBCAM_LABELS_PATH
            
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
    
    def __str__(self):
        return "Office-31 " + self.domain


class DatasetToybox(torchdata.Dataset):
    """
    Class definition for Toybox dataset
    """
    
    def __init__(self, root, train, transform, hypertune, instances, images):
        import dataset_toybox
        self.dataset = dataset_toybox.ToyboxDataset(root=root, train=train, transform=transform, hypertune=hypertune,
                                                    rng=np.random.default_rng(), num_instances=instances,
                                                    num_images_per_class=images)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        return self.dataset[item]
    
    def __str__(self):
        return "Toybox"


class DatasetIN12(torchdata.Dataset):
    """
    Class definition for IN-12 dataset
    """
    
    def __init__(self, root, train, transform, fraction, hypertune):
        import dataset_imagenet12
        self.dataset = dataset_imagenet12.DataLoaderGeneric(root=root, train=train, transform=transform,
                                                            fraction=fraction, hypertune=hypertune)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        return self.dataset[item]
    
    def __str__(self):
        return "IN12"


def get_mean_std(dataset):
    """Return the mean and std of the specified dataset"""
    if dataset == 'amazon':
        mean, std = OFFICE31_AMAZON_MEAN, OFFICE31_AMAZON_STD
    elif dataset == 'dslr':
        mean, std = OFFICE31_DSLR_MEAN, OFFICE31_DSLR_STD
    elif dataset == 'webcam':
        mean, std = OFFICE31_WEBCAM_MEAN, OFFICE31_WEBCAM_STD
    elif dataset == 'toybox':
        mean, std = TOYBOX_MEAN, TOYBOX_STD
    elif dataset == 'in12':
        mean, std = IN12_MEAN, IN12_STD
    elif dataset == "mnist50":
        mean, std = MNIST_MEAN, MNIST_STD
    elif dataset == "svhn-b":
        mean, std = SVHN_MEAN, SVHN_STD
    else:
        raise NotImplementedError("Mean and std for dataset {} has not been specified".format(dataset))
    return mean, std


def get_transform(dataset, mean, std, args):
    """Return the image transform for the specified dataset"""
    if dataset not in DATASETS:
        raise NotImplementedError("Transform for dataset {} not specified...".format(dataset))
    if args['train']:
        trnsform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(size=(224, 224)),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
    else:
        trnsform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
    if dataset == "mnist50":
        trnsform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(32),
                                       transforms.Grayscale(3),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
    if dataset == "svhn-b":
        trnsform = transforms.Compose([transforms.Resize(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
    return trnsform


def get_dataset(d_name, args):
    """Return the class signature of Dataset"""
    if d_name in Office31.DOMAINS:
        return Office31(domain=d_name, transform=args['transform'])
    elif d_name == 'toybox':
        ims = 2000 if args['hypertune'] else 8000
        return DatasetToybox(root=TOYBOX_DATA_PATH, train=args['train'], transform=args['transform'], instances=-1,
                             images=ims, hypertune=args['hypertune'])
    elif d_name == 'in12':
        fr = 0.5 if args['hypertune'] else 1.0
        return DatasetIN12(root=IN12_DATA_PATH, train=args['train'], transform=args['transform'], fraction=fr,
                           hypertune=args['hypertune'])
    elif d_name == 'mnist50':
        return dataset_mnist50.DatasetMNIST50(train=args['train'], transform=args['transform'])
    elif d_name == 'svhn-b':
        return dataset_svhn_balanced.BalancedSVHN(root='../data/', train=args['train'], transform=args['transform'],
                                                  hypertune=args['hypertune'])
    raise NotImplementedError("Transform for dataset {} has not been specified".format(d_name))


def prepare_dataset(d_name, args):
    """Prepare and returh the dataset specified"""
    mean, std = get_mean_std(dataset=d_name)
    
    args['transform'] = get_transform(d_name, mean, std, args=args)
    
    dataset = get_dataset(d_name=d_name, args=args)
    return dataset


if __name__ == "__main__":
    for data in ["mnist50", "svhn-b"]:
        dset = prepare_dataset(data, args={'hypertune': False, "train": True})
        dataloader = torchdata.DataLoader(dset, batch_size=64, shuffle=True)
        print(data, utils.online_mean_and_sd(dataloader))
