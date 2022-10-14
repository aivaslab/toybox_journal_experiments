"""
Module with the networks for the DANN experiments
"""
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    """
    Gradient Reversal module
    """
    @staticmethod
    def forward(ctx, i, alpha):
        """forward method"""
        ctx.alpha = alpha
        return i.view_as(i)
    
    @staticmethod
    def backward(ctx, grad_output):
        """backward method"""
        return grad_output.neg() * ctx.alpha, None


class Network(nn.Module):
    """
    Class definition for the domain adaptation network module
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.classifier = nn.Identity()
        self.domain_classifier = nn.Identity()
    
    def forward(self, x, source_size, alpha):
        """
        Forward method
        """
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        feats_source = feats[:source_size]
        logits = self.classifier(feats_source)
        grad_reversed_out = GradReverse.apply(feats, alpha)
        dom = self.domain_classifier(grad_reversed_out)
        return feats, logits, dom.squeeze()


class SVHNNetwork(Network):
    """
    Network for experiments with SVHN
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.1),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.25),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=1),
                                      nn.Dropout(p=0.25),
                                      )
        self.classifier = nn.Sequential(nn.Linear(512, 3072), nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(3072, 2048), nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(2048, 10)
                                        )
        self.domain_classifier = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(),
                                               nn.Dropout(),
                                               nn.Linear(1024, 1024), nn.ReLU(),
                                               nn.Dropout(),
                                               nn.Linear(1024, 1)
                                               )


class MNISTNetwork(Network):
    """
    Network for MNIST experiments
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      )
        self.classifier = nn.Sequential(nn.Linear(1200, 100), nn.ReLU(),
                                        nn.Linear(100, 100), nn.ReLU(),
                                        nn.Linear(100, 10)
                                        )
        self.domain_classifier = nn.Sequential(nn.Linear(1200, 100), nn.ReLU(),
                                               nn.Linear(100, 1)
                                               )
        

class ToyboxNetwork(Network):
    """
    Network for Toybox experiments
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
        self.domain_classifier = nn.Sequential(nn.Linear(self.fc_size, 1000), nn.ReLU(), nn.Linear(1000, 1))
        
        
def get_network(source_dataset, target_dataset):
    """
    Returns the preferred network for the specified datasets
    :return:
    """
    if source_dataset == 'svhn' or target_dataset == 'svhn':
        return SVHNNetwork()
    elif 'mnist' in source_dataset and 'mnist' in target_dataset:
        return MNISTNetwork()
    elif 'toybox' in source_dataset or 'toybox' in target_dataset:
        return ToyboxNetwork()
    raise NotImplementedError("Network for src: {] and trgt: {} not implemented...".format(source_dataset,
                                                                                           target_dataset))
    

