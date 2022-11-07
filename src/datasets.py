"""
Module with datasets for DANN, JAN and SE methods
"""
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import numpy as np

import self_ensemble_aug
import dataset_mnist50
import dataset_svhn_balanced


MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)

SVHN_MEAN = (0.4377, 0.4437, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

IN12_MEAN = (0.4541, 0.4845, 0.4980)
IN12_STD = (0.2928, 0.2738, 0.2756)

TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 0.1894, 0.1775)

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"


def get_mean_std(dataset):
    """Return the mean and std of the specified dataset"""
    if dataset == 'mnist' or dataset == 'mnist50':
        mean, std = MNIST_MEAN, MNIST_STD
    elif dataset == 'svhn' or dataset == 'svhn-b':
        mean, std = SVHN_MEAN, SVHN_STD
    elif dataset == 'toybox':
        mean, std = TOYBOX_MEAN, TOYBOX_STD
    elif dataset == 'in12':
        mean, std = IN12_MEAN, IN12_STD
    else:
        raise NotImplementedError("Mean and std for dataset {} has not been specified".format(dataset))
    return mean, std


def get_transform(dataset, mean, std, train):
    """Return the image transform for the specified dataset"""
    if dataset == 'mnist50':
        trnsfrms = [transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)]
    elif dataset == 'svhn-b':
        trnsfrms = [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)]
    elif dataset == 'toybox':
        if train:
            trnsfrms = [transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(size=224),
                        transforms.ColorJitter(hue=0.3, contrast=0.5, saturation=0.5, brightness=0.3),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)
                        ]
        else:
            trnsfrms = [transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)
                        ]
    elif dataset == "in12":
        if train:
            trnsfrms = [transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(size=224),
                        transforms.ColorJitter(hue=0.5, contrast=0.5, saturation=0.4, brightness=0.3),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
                        ]
        else:
            trnsfrms = [transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)]
    else:
        raise NotImplementedError("Transform for dataset {} not specified...".format(dataset))
    transform = transforms.Compose(trnsfrms)
    return transform


def get_dataset(d_name, args):
    """Return the appropriate Dataset class"""
    if d_name == 'mnist50':
        return DatasetMNIST50(train=args['train'], transform=args['transform'], special_aug=args['special_aug'])
    
    if d_name == 'svhn-b':
        if args['pair']:
            return DatasetSVHNBPair(train=args['train'], transform=args['transform'], hypertune=args['hypertune'],
                                    special_aug=args['special_aug'])
        return DatasetSVHNB(train=args['train'], transform=args['transform'], hypertune=args['hypertune'],
                            special_aug=args['special_aug'])
    
    if d_name == 'in12':
        fr = 0.5 if args['hypertune'] else 1.0
        if args['pair']:
            return DatasetIN12Pair(root=IN12_DATA_PATH, train=args['train'], transform=args['transform'], fraction=fr,
                                   hypertune=args['hypertune'])
        return DatasetIN12(root=IN12_DATA_PATH, train=args['train'], transform=args['transform'], fraction=fr,
                           hypertune=args['hypertune'])
    if d_name == "toybox":
        ims = 2000 if args['hypertune'] else 8000
        return DatasetToybox(root=TOYBOX_DATA_PATH, train=args['train'], transform=args['transform'], instances=-1,
                             images=ims, hypertune=args['hypertune'])
    
    raise NotImplementedError("Transform for dataset {} has not been specified".format(d_name))


def prepare_dataset(d_name, args):
    """Prepare and returh the dataset specified"""
    mean, std = get_mean_std(dataset=d_name)
    args['transform'] = get_transform(d_name, mean, std, train=args['train'])
    
    dataset = get_dataset(d_name=d_name, args=args)
    return dataset


class DatasetMNIST50(torchdata.Dataset):
    """
    Class definition for MNIST50
    """
    
    def __init__(self, train, transform, special_aug):
        self.train = train
        self.special_aug = special_aug
        self.aug = self_ensemble_aug.get_aug_for_mnist()
        self.dataset = dataset_mnist50.DatasetMNIST50(train=train, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        idx, img, label = self.dataset[item]
        if self.special_aug is True and self.train is True:
            img = self.aug.augment(img.unsqueeze(0))
        return idx, img.squeeze(), label
    
    def __str__(self):
        return "MNIST50"


class DatasetSVHNB(torchdata.Dataset):
    """
    Class definition for BalancedSVHN dataset
    """
    
    def __init__(self, train, transform, hypertune, special_aug):
        self.train = train
        self.special_aug = special_aug
        self.aug = self_ensemble_aug.get_aug_for_mnist()
        self.transform = transform
        self.hypertune = hypertune
        self.dataset = dataset_svhn_balanced.BalancedSVHN(root='../data/', train=train, transform=transform,
                                                          hypertune=hypertune)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        idx, img, label = self.dataset[item]
        if self.special_aug is True and self.train is True:
            img = self.aug.augment(img.unsqueeze(0))
        return idx, img.squeeze(), label
    
    def __str__(self):
        return "Balanced SVHN"


class DatasetSVHNBPair(torchdata.Dataset):
    """
    Dataset Class for SVHN
    """
    
    def __init__(self, train, transform, hypertune, special_aug):
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.special_aug = special_aug
        self.dataset = dataset_svhn_balanced.BalancedSVHN(root='../data/', train=self.train, transform=None,
                                                          hypertune=self.hypertune)
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        item, img, label = self.dataset[item]
        img1, img2 = self.transform(img), self.transform(img)
        if self.special_aug and self.train:
            img1, img2 = self.aug.augment(img1.unsqueeze(0)), self.aug.augment(img2.unsqueeze(0))
            img1, img2 = img1.squeeze(), img2.squeeze()
        return item, (img1, img2), label
    
    def __str__(self):
        return "Balanced SVHN Pair"


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


class DatasetIN12Pair(torchdata.Dataset):
    """
    Class definition for IN-12 dataset
    """
    
    def __init__(self, root, train, transform, fraction, hypertune):
        self.transform = transform
        import dataset_imagenet12
        self.dataset = dataset_imagenet12.DataLoaderGeneric(root=root, train=train, transform=None,
                                                            fraction=fraction, hypertune=hypertune)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        idx, img, label = self.dataset[item]
        img1, img2 = self.transform(img), self.transform(img)
        return idx, (img1, img2), label
    
    def __str__(self):
        return "IN12 Pair"
