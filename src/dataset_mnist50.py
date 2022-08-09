"""
Dataset class for MNIST-50
"""
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import numpy as np


class DatasetMNIST50(torchdata.Dataset):
    """
    Dataset class for MNIST
    """
    TRAIN_IMAGES_FILE = "../data/mnist50_train_images.npy"
    TRAIN_LABELS_FILE = "../data/mnist50_train_labels.npy"
    VAL_IMAGES_FILE = "../data/mnist50_val_images.npy"
    VAL_LABELS_FILE = "../data/mnist50_val_labels.npy"
    
    def __init__(self, root, train, transform):
        super(DatasetMNIST50, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        
        if self.train:
            self.images = np.load(self.TRAIN_IMAGES_FILE)
            self.labels = np.load(self.TRAIN_LABELS_FILE)
        else:
            self.images = np.load(self.VAL_IMAGES_FILE)
            self.labels = np.load(self.VAL_LABELS_FILE)
    
    def __getitem__(self, item):
        img, label = self.images[item], self.labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return item, img, label
    
    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.Grayscale(3),
        transforms.RandomInvert(p=0.9),
    ])
    dataset = DatasetMNIST50(train=True, transform=tr, root="")
    print(len(dataset))
    idx, img, lbl = dataset[499]
    img.show()
