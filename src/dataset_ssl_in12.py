"""
This module implements the IN-12 dataset for SimCLR-like methods
"""
import torch
import pickle
import cv2
import numpy as np


class DatasetIN12(torch.utils.data.Dataset):
    """
    This class can be used to load IN-12 data in PyTorch for SimCLR/BYOL-like methods
    """
    def __init__(self, fraction=0.1, transform=None, hypertune=True):
        self.fraction = fraction
        self.transform = transform
        self.hypertune = hypertune
        super().__init__()
        if self.hypertune:
            self.train_images_file_name = "../data_12/IN-12/dev.pickle"
        else:
            self.train_images_file_name = "../data_12/IN-12/train.pickle"
        
        with open(self.train_images_file_name, "rb") as train_images_file:
            self.train_images = pickle.load(train_images_file)
            
        num_images_all = len(self.train_images)
        num_images_train = int(self.fraction * num_images_all)
        rng = np.random.default_rng(0)
        self.selected_indices = rng.choice(num_images_all, num_images_train, replace=False)
        
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, index):
        actual_index = self.selected_indices[index]
        img = np.array(cv2.imdecode(self.train_images[actual_index], 3))
        if self.transform is not None:
            imgs = [self.transform(img) for _ in range(2)]
        else:
            imgs = [img, img]
            
        return index, imgs
    

if __name__ == "__main__":
    import torchvision.transforms as transforms

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.2, saturation=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5),
        transforms.Resize(256),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
    ])
    train_data = DatasetIN12(fraction=0.1, transform=train_transform)
    print(len(train_data))
    for i in range(5):
        idx, images = train_data[i]
        im = transforms.ToPILImage()(images[0])
        im.show()
        im2 = transforms.ToPILImage()(images[1])
        im2.show()
