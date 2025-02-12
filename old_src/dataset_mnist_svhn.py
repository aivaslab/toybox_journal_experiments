"""
Datasets for MNIST and SVHN which match dataset format for IN-12 and Toybox
"""
import torchvision.datasets as datasets
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch
import numpy as np

MNIST_MEAN = (0.1309, 0.1309, 0.1309)
MNIST_STD = (0.2893, 0.2893, 0.2893)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


class DatasetMNIST(torchdata.Dataset):
    """
    Dataset class for MNIST
    """
    
    def __init__(self, root, train, transform, hypertune=True):
        super(DatasetMNIST, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        
        if self.hypertune:
            self.dataset = datasets.MNIST(root=self.root, train=True, transform=self.transform, download=True)
            rng = np.random.default_rng(seed=42)
            all_images = len(self.dataset)
            all_indices = np.arange(all_images)
            train_num = int(0.8 * all_images)
            self.train_indices = rng.choice(all_indices, train_num, replace=False)
            self.test_indices = [i for i in all_indices if i not in self.train_indices]
        else:
            self.dataset = datasets.MNIST(root=self.root, train=self.train, transform=self.transform, download=True)
    
    def __getitem__(self, item):
        if self.hypertune:
            if self.train:
                actual_item = self.train_indices[item]
            else:
                actual_item = self.test_indices[item]
        else:
            actual_item = item
        img, label = self.dataset[actual_item]
        return actual_item, img, label
    
    def __len__(self):
        if self.hypertune:
            if self.train:
                return len(self.train_indices)
            else:
                return len(self.test_indices)
        else:
            return len(self.dataset)


class DatasetSVHN(torchdata.Dataset):
    """
    Dataset class for SVHN
    """
    
    def __init__(self, root, train, transform, hypertune=True):
        super(DatasetSVHN, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        if self.hypertune or self.train:
            self.split = 'train'
        else:
            self.split = 'test'
        
        if self.hypertune:
            self.dataset = datasets.SVHN(root=self.root, split=self.split, transform=self.transform, download=True)
            rng = np.random.default_rng(seed=42)
            all_images = len(self.dataset)
            all_indices = np.arange(all_images)
            train_num = int(0.8 * all_images)
            self.train_indices = rng.choice(all_indices, train_num, replace=False)
            self.test_indices = [i for i in all_indices if i not in self.train_indices]
        else:
            self.dataset = datasets.SVHN(root=self.root, split=self.split, transform=self.transform, download=True)
    
    def __getitem__(self, item):
        if self.hypertune:
            if self.train:
                actual_item = self.train_indices[item]
            else:
                actual_item = self.test_indices[item]
        else:
            actual_item = item
        img, label = self.dataset[actual_item]
        return actual_item, img, label
    
    def __len__(self):
        if self.hypertune:
            if self.train:
                return len(self.train_indices)
            else:
                return len(self.test_indices)
        else:
            return len(self.dataset)


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    
    for _, data, _ in loader:
        # print(data.shape)
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        
        cnt += nb_pixels
    
    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == "__main__":
    tr = transforms.Compose([transforms.Grayscale(3), transforms.Pad(2),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
                             # transforms.ToTensor(),
                             # transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD)
                             ])
    dataset = DatasetMNIST(root="../data/", train=True, hypertune=True, transform=tr)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    # print(online_mean_and_sd(dataloader))
    import visda_aug
    
    hflip = False
    xlat_range = 2
    affine_std = 0.1
    intens_scale_range_lower = 0.25
    intens_scale_range_upper = 0.5
    intens_offset_range_lower = -0.5
    intens_offset_range_upper = 0.5
    intens_flip = True
    gaussian_noise_std = 0.1

    aug = visda_aug.ImageAugmentation(hflip, xlat_range, affine_std,
                                      intens_scale_range_lower=intens_scale_range_lower,
                                      intens_scale_range_upper=intens_scale_range_upper,
                                      intens_offset_range_lower=intens_offset_range_lower,
                                      intens_offset_range_upper=intens_offset_range_upper,
                                      intens_flip=intens_flip, gaussian_noise_std=gaussian_noise_std)
    
    indices, images, labels = next(iter(dataloader))
    print(images.shape)
    images = aug.augment(images)
    print(type(images), images.shape)
    for idx in range(10):
        img = images[idx]
        unnorm = visda_aug.UnNormalize(mean=MNIST_MEAN, std=MNIST_STD)
        img = unnorm(img)
        img = transforms.ToPILImage()(img)
        img.show()
    # count = [0] * 10
    # for idx, images, labels in dataloader:
    #     for label in labels:
    #         count[label] += 1
    # print(count)
    # total = sum(count)
    # count = [cnt/total for cnt in count]
    # print(count)
