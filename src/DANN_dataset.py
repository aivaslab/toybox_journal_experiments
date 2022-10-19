"""
Datasets for the DANN experiments
"""
import torch.utils.data as torchdata
import pickle
import csv
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import utils

UNIT_STD = (1.0, 1.0, 1.0)

MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)

SVHN_MEAN = (0.4377, 0.4437, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

MNISTM_MEAN = (0.4082, 0.4621, 0.4579)
MNISTM_STD = (0.2587, 0.2368, 0.2519)

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)

TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 0.1894, 0.1775)

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"


def get_mean_std(dataset):
    """Return the mean and std of the specified dataset"""
    if dataset == 'mnist':
        mean, std = MNIST_MEAN, MNIST_STD
    elif dataset == 'mnist-m':
        mean, std = MNISTM_MEAN, MNISTM_STD
    elif dataset == 'svhn':
        mean, std = SVHN_MEAN, SVHN_STD
    elif dataset == 'toybox':
        mean, std = TOYBOX_MEAN, TOYBOX_STD
    elif dataset == 'in12':
        mean, std = IN12_MEAN, IN12_STD
    else:
        raise NotImplementedError("Mean and std for dataset {} has not been specified".format(dataset))
    return mean, std


def get_transform(dataset, mean, std):
    """Return the image transform for the specified dataset"""
    trnsfrms = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if dataset == 'mnist':
        trnsfrms.insert(0, transforms.Grayscale(3))
    elif dataset == 'mnist-m':
        trnsfrms.insert(0, transforms.ToPILImage())
    elif dataset == 'svhn':
        pass  # no modification required for svhn dataset
    elif dataset == 'toybox':
        trnsfrms = [transforms.ToPILImage(),
                    transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0.5, brightness=0.3),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)
                    ]
    elif dataset == "in12":
        trnsfrms = [transforms.ToPILImage(),
                    transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0, brightness=0.3),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
                    ]
    else:
        raise NotImplementedError("Transform for dataset {} not specified...".format(dataset))
    transform = transforms.Compose(trnsfrms)
    return transform


def get_dataset(d_name, args):
    """Return the class signature of Dataset"""
    if d_name == 'mnist':
        return DatasetMNIST(transform=args['transform'], train=args['train'])
    if d_name == 'svhn':
        return DatasetSVHN(transform=args['transform'], train=args['train'])
    if d_name == 'mnist-m':
        return DatasetMNISTM(train=args['train'], transform=args['transform'])
    if d_name == 'in12':
        fr = 0.2 if args['hypertune'] else 1.0
        return DatasetIN12(root=IN12_DATA_PATH, train=args['train'], transform=args['transform'], fraction=fr,
                           hypertune=args['hypertune'])
    if d_name == "toybox":
        ims = 500 if args['hypertune'] else 5000
        return DatasetToybox(root=TOYBOX_DATA_PATH, train=args['train'], transform=args['transform'], instances=-1,
                             images=ims, hypertune=args['hypertune'])
    
    raise NotImplementedError("Transform for dataset {} has not been specified".format(d_name))


def prepare_dataset(d_name, args):
    """Prepare and returh the dataset specified"""
    mean, std = get_mean_std(dataset=d_name)

    if not args['normalize']:
        std = UNIT_STD
    args['transform'] = get_transform(d_name, mean, std)
    
    dataset = get_dataset(d_name=d_name, args=args)
    return dataset


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


class DatasetMNISTM(torchdata.Dataset):
    """
    MNIST-M Dataset
    """
    
    TRAIN_IMAGES_FILE = "../../DATASETS/mnist_m/mnist_m_train.pkl"
    TRAIN_LABELS_FILE = "../../DATASETS/mnist_m/mnist_m_train.csv"
    TEST_IMAGES_FILE = "../../DATASETS/mnist_m/mnist_m_test.pkl"
    TEST_LABELS_FILE = "../../DATASETS/mnist_m/mnist_m_test.csv"
    
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        
        if self.train:
            images_file_name = self.TRAIN_IMAGES_FILE
            labels_file_name = self.TRAIN_LABELS_FILE
        else:
            images_file_name = self.TEST_IMAGES_FILE
            labels_file_name = self.TEST_LABELS_FILE
        
        images_file = open(images_file_name, "rb")
        labels_file = open(labels_file_name, "r")
        self.images = pickle.load(images_file)
        self.data = list(csv.DictReader(labels_file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        img, label = np.array(cv2.imdecode(self.images[item], 3)), int(self.data[item]['Label'])
        if self.transform is not None:
            img = self.transform(img)
        return item, img, label
    
    
class DatasetMNIST(torchdata.Dataset):
    """
    Dataset Class for MNIST
    """
    def __init__(self, train, transform):
        self.train = train
        self.transform = transform
        self.data = datasets.MNIST(root="../data/", train=self.train, transform=self.transform, download=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        img, label = self.data[item]
        return item, img, label


class DatasetSVHN(torchdata.Dataset):
    """
    Dataset Class for SVHN
    """
    
    def __init__(self, train, transform):
        self.train = train
        self.split = 'train' if self.train else 'test'
        self.transform = transform
        self.data = datasets.SVHN(root="../data/", split=self.split, transform=self.transform, download=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        img, label = self.data[item]
        return item, img, label
