"""Datasets for the ATT algorithm"""
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import torchvision.datasets as datasets
import numpy as np

import self_ensemble_aug

MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)

SVHN_MEAN = (0.4377, 0.4437, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def get_mean_std(dataset):
    """Return the mean and std of the specified dataset"""
    if dataset == 'mnist':
        mean, std = MNIST_MEAN, MNIST_STD
    elif dataset == 'svhn':
        mean, std = SVHN_MEAN, SVHN_STD
    else:
        raise NotImplementedError("Mean and std for dataset {} has not been specified".format(dataset))
    return mean, std


def get_transform(dataset, mean, std):
    """Return the image transform for the specified dataset"""
    trnsfrms = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if dataset == 'mnist':
        trnsfrms.insert(0, transforms.Grayscale(3))
    elif dataset == 'svhn':
        pass  # no modification required for svhn dataset
    else:
        raise NotImplementedError("Transform for dataset {} not specified...".format(dataset))
    transform = transforms.Compose(trnsfrms)
    return transform


def get_dataset(d_name, args):
    """Return the class signature of Dataset"""
    if d_name == 'mnist':
        return DatasetMNIST(transform=args['transform'], train=args['train'], special_aug=args['special_aug'],
                            indices=args['indices'])
    if d_name == 'svhn':
        return DatasetSVHN(transform=args['transform'], train=args['train'], special_aug=args['special_aug'],
                           indices=args['indices'])
    raise NotImplementedError("Transform for dataset {} has not been specified".format(d_name))


def prepare_dataset(d_name, args):
    """Prepare and returh the dataset specified"""
    mean, std = get_mean_std(dataset=d_name)
    
    args['transform'] = get_transform(d_name, mean, std)
    args['special_aug'] = False
    
    dataset = get_dataset(d_name=d_name, args=args)
    return dataset


class DatasetMNIST(torchdata.Dataset):
    """
    Dataset Class for MNIST
    """
    
    def __init__(self, train, transform, special_aug, indices=None):
        self.indices = indices
        self.train = train
        self.transform = transform
        self.data = datasets.MNIST(root="../data/", train=self.train, transform=self.transform, download=True)
        if self.indices is None:
            self.indices = np.arange(len(self.data))
        self.special_aug = special_aug
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item):
        index = self.indices[item]
        img, label = self.data[index]
        if self.special_aug and self.train is True:
            img = self.aug.augment(img.unsqueeze(0))
        return item, img.squeeze(), label
    
    def __str__(self):
        return "MNIST " + str(len(self.indices))


class DatasetSVHN(torchdata.Dataset):
    """
    Dataset Class for SVHN
    """
    
    def __init__(self, train, transform, special_aug, indices=None):
        self.indices = indices
        self.train = train
        self.split = 'train' if self.train else 'test'
        self.transform = transform
        self.data = datasets.SVHN(root="../data/", split=self.split, transform=self.transform, download=True)
        if self.indices is None:
            self.indices = np.arange(len(self.data))
        self.special_aug = special_aug
        self.aug = self_ensemble_aug.get_aug_for_mnist()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item):
        index = self.indices[item]
        img, label = self.data[index]
        if self.special_aug and self.train is True:
            img = self.aug.augment(img.unsqueeze(0))
        return item, img.squeeze(), label
    
    def __str__(self):
        return "SVHN " + str(len(self.indices))
    
    
def check_datasets():
    """Check if datasets are working correctly"""
    dataset2_args = {'train': True, 'indices': None}
    dataset2 = prepare_dataset('svhn', args=dataset2_args)
    print(len(dataset2))
    indices = np.random.choice(len(dataset2), 1000, replace=False)
    # print(indices)
    
    dataset_args = {'train': True, 'indices': indices}
    dataset = prepare_dataset('svhn', args=dataset_args)
    print(len(dataset))
    
    mean, std = SVHN_MEAN, SVHN_STD
    inverse_trnsfrm = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]),
        # transforms.ToPILImage()
    ])
    
    for ind in range(len(indices)):
        idx, img, lbl = dataset[ind]
        img = inverse_trnsfrm(img).numpy()
        
        idx2, img2, lbl2 = dataset2[indices[ind]]
        img2 = inverse_trnsfrm(img2).numpy()
        c, x, y = img2.squeeze().shape
        for i in range(x):
            for j in range(y):
                for k in range(c):
                    assert (img[k][i][j] == img2[k][i][j])


if __name__ == "__main__":
    check_datasets()
    