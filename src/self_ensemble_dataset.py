"""
Datasets for the Self-Ensemble experiments
"""
import torch.utils.data as torchdata
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import utils
import dataset_mnist50
import dataset_svhn_balanced
import self_ensemble_aug

UNIT_STD = (1.0, 1.0, 1.0)

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


def get_transform(dataset, mean, std):
    """Return the image transform for the specified dataset"""
    trnsfrms = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if dataset == 'mnist':
        trnsfrms.insert(0, transforms.Grayscale(3))
    elif dataset == 'mnist50':
        trnsfrms = [transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)]
    elif dataset == 'svhn':
        pass  # no modification required for svhn dataset
    elif dataset == 'svhn-b':
        trnsfrms = [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)]
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
    
    if d_name == 'mnist50':
        return DatasetMNIST50(train=args['train'], transform=args['transform'])
    
    if d_name == 'svhn':
        if args['target']:
            return DatasetSVHNPair(transform=args['transform'], train=args['train'])
        return DatasetSVHN(transform=args['transform'], train=args['train'])
    
    if d_name == 'svhn-b':
        return DatasetSVHNB(train=args['train'], transform=args['transform'], hypertune=args['hypertune'])
    if d_name == 'in12':
        fr = 0.5 if args['hypertune'] else 1.0
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


class DatasetMNIST50(torchdata.Dataset):
    """
    Dataset Class for MNIST
    """
    
    def __init__(self, train, transform):
        self.train = train
        self.transform = transform
        self.dataset = dataset_mnist50.DatasetMNIST50(train=self.train, transform=self.transform)
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        item, img, label = self.dataset[item]
        if self.train:
            img = self.aug.augment((img.unsqueeze(0)))
        return item, img.squeeze(), label
    
    def __str__(self):
        return "MNIST"


class DatasetSVHNB(torchdata.Dataset):
    """
    Dataset Class for SVHN
    """
    
    def __init__(self, train, transform, hypertune):
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.dataset = dataset_svhn_balanced.BalancedSVHN(root='../data/', train=self.train, transform=self.transform,
                                                          hypertune=self.hypertune)
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        item, img, label = self.data[item]
        if False and self.train:
            img = self.aug.augment(img.unsqueeze(0))
        return item, img.squeeze(), label
    
    def __str__(self):
        return "Balanced SVHN"


class DatasetMNIST(torchdata.Dataset):
    """
    Dataset Class for MNIST
    """
    
    def __init__(self, train, transform):
        self.train = train
        self.transform = transform
        self.data = datasets.MNIST(root="../data/", train=self.train, transform=self.transform, download=True)
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        img, label = self.data[item]
        if self.train:
            img = self.aug.augment((img.unsqueeze(0)))
        return item, img.squeeze(), label
    
    def __str__(self):
        return "MNIST"


class DatasetSVHN(torchdata.Dataset):
    """
    Dataset Class for SVHN
    """
    
    def __init__(self, train, transform):
        self.train = train
        self.split = 'train' if self.train else 'test'
        self.transform = transform
        self.data = datasets.SVHN(root="../data/", split=self.split, transform=self.transform, download=True)
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        img, label = self.data[item]
        if self.train:
            img = self.aug.augment(img.unsqueeze(0))
        return item, img.squeeze(), label
    
    def __str__(self):
        return "SVHN"


class DatasetSVHNPair(torchdata.Dataset):
    """
    Dataset Class for SVHN. Returns two images instead of one
    """
    
    def __init__(self, train, transform):
        self.train = train
        self.split = 'train' if self.train else 'test'
        self.transform = transform
        self.data = datasets.SVHN(root="../data/", split=self.split, transform=None, download=True)
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        img, label = self.data[item]
        img1, img2 = self.transform(img), self.transform(img)
        img1, img2 = self.aug.augment(img1.unsqueeze(0)), self.aug.augment(img2.unsqueeze(0))
        img1, img2 = img1.squeeze(), img2.squeeze()
        return item, (img1, img2), label
    
    def __str__(self):
        return "SVHN Pair"


def main():
    """Main method call to avoid warnings"""
    dataset = prepare_dataset(d_name='svhn', args={'train': True, 'hypertune': True, 'target': False})
    dataloader = torchdata.DataLoader(dataset, batch_size=64, shuffle=False)
    _, images, _ = next(iter(dataloader))
    print(type(images), images.shape)
    ims = utils.get_images(images=images, mean=SVHN_MEAN, std=SVHN_STD, aug=True)
    ims.show()

    dataset = prepare_dataset(d_name='mnist', args={'train': True, 'hypertune': True, 'target': False})
    dataloader = torchdata.DataLoader(dataset, batch_size=64, shuffle=False)
    _, images, _ = next(iter(dataloader))
    print(type(images), images.shape)
    ims = utils.get_images(images=images, mean=MNIST_MEAN, std=MNIST_STD, aug=True)
    ims.show()


if __name__ == "__main__":
    main()
    