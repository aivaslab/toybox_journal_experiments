"""
This module implements the Domain Adaptation by Backpropagation paper by Ganin and Lampitsky
"""
import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as torchdata
import torch.nn as nn
from torch.autograd import Function
import tqdm


class GradReverse(Function):
    """
    Gradient Reversal method
    """
    @staticmethod
    def forward(ctx, i):
        """forward method"""
        return i.view_as(i)
    
    @staticmethod
    def backward(ctx, grad_output):
        """backward method"""
        return grad_output * -1
    
    
def grad_reverse(x):
    """sdf"""
    return GradReverse.apply(x)


class Network(nn.Module):
    """
    Class definition for the domain adaptation module
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4))
                                      )
        self.classifier = nn.Sequential(nn.Linear(128, 3072), nn.ReLU(),
                                        nn.Linear(3072, 2048), nn.ReLU(),
                                        nn.Linear(2048, 10)
                                        )
        self.domain_classifier = nn.Sequential(nn.Linear(128, 1024), nn.ReLU(),
                                               nn.Linear(1024, 1024), nn.ReLU(),
                                               nn.Linear(1024, 1)
                                               )
        
    def forward(self, x, source_size):
        """
        Forward method
        """
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        feats_source = feats[:source_size]
        logits = self.classifier(feats_source)
        grad_reversed_out = grad_reverse(feats)
        dom = self.domain_classifier(grad_reversed_out)
        return feats, logits, dom.squeeze()
    
    
class Experiment:
    """
    Class used to run the experiments
    """
    def __init__(self):
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                        transforms.Normalize()])
        self.dataset1 = datasets.MNIST(root="../data/", train=True)
    
    
if __name__ == "__main__":
    net = Network()
    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=0.01, weight_decay=1e-5)
    optimizer.add_param_group({'params': net.classifier.parameters()})
    optimizer.add_param_group({'params': net.domain_classifier.parameters()})
    mnist_transform = transforms.Compose([transforms.Grayscale(3),
                                          transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0.5, brightness=0.3),
                                          transforms.Resize(32),
                                          transforms.ToTensor(),
                                          # transforms.Normalize(mean=(0.1307, 0.1307, 0.1307),
                                          #                      std=(0.3081, 0.3081, 0.3081))
                                          ])
    
    svhn_transform = transforms.Compose([transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0.5, brightness=0.3),
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         # transforms.Normalize(mean=(0.4377, 0.4437, 0.4728),
                                         #                      std=(0.1980, 0.2010, 0.1970))
                                         ])
    
    dataset1 = datasets.MNIST(root="../data/", train=True, transform=mnist_transform, download=True)
    dataset2 = datasets.SVHN(root="../data/", split='train', download=True, transform=svhn_transform)
    
    loader_1 = torchdata.DataLoader(dataset1, batch_size=128, shuffle=True, num_workers=4)
    loader_2 = torchdata.DataLoader(dataset2, batch_size=128, shuffle=True, num_workers=4)
    loader_2_iter = iter(loader_2)
    net = net.cuda()
    num_epochs = 2
    for ep in range(1, num_epochs + 1):
        tqdm_bar = tqdm.tqdm(loader_1)
        for img1, labels1 in tqdm_bar:
            try:
                img2, labels2 = next(loader_2_iter)
            except StopIteration:
                loader_2_iter = iter(loader_2)
                img2, labels2 = next(loader_2_iter)
            optimizer.zero_grad()
            images = torch.concat([img1, img2], dim=0)
            b_size = images.size(0)
            dom_labels = torch.cat([torch.zeros(img1.size(0)), torch.ones(img2.size(0))])
            images, dom_labels, labels1 = images.cuda(), dom_labels.cuda(), labels1.cuda()
            
            f, l, d = net.forward(images, img1.size(0))
            ce_loss = nn.CrossEntropyLoss()(l, labels1)
            
            dom_pred = torch.sigmoid(d)
            dom_loss = nn.BCELoss()(dom_pred, dom_labels)
            
            total_loss = ce_loss + dom_loss
            total_loss.backward()
            tqdm_bar.set_description("Ep: {}/{}  CE Loss: {:.4f}  Dom Loss: {:.4f}".format(ep, num_epochs,
                                                                                           ce_loss.item(),
                                                                                           dom_loss.item()))
            optimizer.step()
