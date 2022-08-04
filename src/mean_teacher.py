"""
Module for implementing the Mean Teacher model from the VisDA-2017 winning submission.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
import torch.utils.data as torchdata
import torch.nn.functional as F

import dataset_mnist_svhn

MNIST_MEAN = (0.1309, 0.1309, 0.1309)
MNIST_STD = (0.2893, 0.2893, 0.2893)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


class Network(nn.Module):
    """
    Backbone network for mean teacher model
    Code sourced from: https://github.com/Britefury/self-ensemble-visual-domain-adapt
    """
    
    def __init__(self, n_classes):
        super().__init__()
    
        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()
    
        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()
    
        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = nn.BatchNorm2d(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = nn.BatchNorm2d(128)
    
        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        """Forward prop for the network"""
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        x = self.drop1(x)
    
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        x = self.drop2(x)
    
        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x)))
    
        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)
    
        x = self.fc4(x)
        return x


class MeanTeacher:
    """
    Module for implementing the mean teacher architecture
    """
    def __init__(self):
        self.student = Network(n_classes=10)
        self.teacher = Network(n_classes=10)
        
        self.mnist_train_transform = transforms.Compose([transforms.Grayscale(3),
                                                         transforms.Resize(32),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
        self.svhn_train_transform = transforms.Compose([transforms.Resize(32),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD)])
        
        self.source_dataset = dataset_mnist_svhn.DatasetMNIST(root="../data/", train=True, hypertune=True,
                                                              transform=self.mnist_train_transform)
        self.target_dataset = dataset_mnist_svhn.DatasetSVHN(root="./data/", train=True, hypertune=True,
                                                             transform=self.svhn_train_transform)
        self.source_loader = torchdata.DataLoader(self.source_dataset, batch_size=256, shuffle=True, num_workers=4,
                                                  pin_memory=True, persistent_workers=True)
        self.target_loader = torchdata.DataLoader(self.target_dataset, batch_size=256, shuffle=True, num_workers=4,
                                                  pin_memory=True, persistent_workers=True)
        
    def update_teacher(self):
        """EMA update for the teacher's weights"""
        for current_params, moving_params in zip(self.teacher.parameters(), self.student.parameters()):
            current_weight, moving_weight = current_params.data, moving_params.data
            current_params.data = current_weight * 0.999 + moving_weight * 0.001
            
    def train(self):
        """Train the network"""
        for params in self.student.parameters():
            params.requires_grad = True
        for params in self.teacher.parameters():
            params.requires_grad = False

    
class Experiment:
    """
    Class to set up and run MeanTeacher experiments
    """
    def run(self):
        """Run the experiment with the specified parameters"""
        pass
    
    
if __name__ == "__main__":
    exp = Experiment()
    exp.run()
