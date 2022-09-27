"""
Module with dataset class for CoRE50 dataset
"""
import torch.utils.data
import pickle
import csv
import cv2
import numpy as np
import torchvision.transforms as transforms

CORE50_MEAN = (0.5980, 0.5622, 0.5363)
CORE50_STD = (0.2102, 0.2199, 0.2338)


class DatasetCoRE50(torch.utils.data.Dataset):
    """
    Dataset class for CoRE50 dataset
    """
    
    def __init__(self, root, train, transform=None, fraction=0.1):
        self.root = root
        self.train = train
        self.transform = transform
        self.fraction = fraction
        self.rng = np.random.default_rng(0)
        
        if self.train:
            self.images_file = self.root + "/core50_data_train_object.pickle"
            self.labels_file = self.root + "/core50_data_train_object.csv"
        else:
            self.images_file = self.root + "/core50_data_test_object.pickle"
            self.labels_file = self.root + "/core50_data_test_object.csv"
        
        with open(self.images_file, "rb") as images_pickle_file:
            self.data = pickle.load(images_pickle_file)
            self.selected_indices = self.rng.choice(len(self.data), int(self.fraction * len(self.data)), replace=False)
            assert len(self.selected_indices) == len(set(self.selected_indices))
        with open(self.labels_file, "r") as labels_file:
            self.labels = list(csv.DictReader(labels_file))
        
    def __getitem__(self, index):
        if self.train:
            actual_index = self.selected_indices[index]
        else:
            actual_index = index
            
        img = np.array(cv2.imdecode(self.data[actual_index], 3))
        label = int(self.labels[actual_index]['Class ID'])
        if self.transform is not None:
            img = self.transform(img)
        return index, img, label
    
    def __len__(self):
        if self.train:
            return len(self.selected_indices)
        else:
            return len(self.data)


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
    tr = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=CORE50_MEAN, std=CORE50_STD)
                             ])
    dataset = DatasetCoRE50(root="../../toybox-representation-learning/data/", transform=tr, train=False, fraction=0.1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    print(len(dataset))
    print(online_mean_and_sd(dataloader))
