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
        
    def forward(self, x):
        """
        Forward method
        """
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.classifier(feats)
        grad_reversed_out = grad_reverse(feats)
        dom = self.domain_classifier(grad_reversed_out)
        return feats, logits, dom
    
    
if __name__ == "__main__":
    data = torch.randn((5, 3, 32, 32))
    net = Network()
    f, l, d = net.forward(data)
    print(f.shape, d.shape, l.shape)
    loss = nn.CrossEntropyLoss()(l, torch.zeros(5).long())
    loss += torch.mean(d)
    loss.backward()
    