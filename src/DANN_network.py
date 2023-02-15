"""
Module with the networks for the DANN experiments
"""
import torch.nn as nn
from torch.autograd import Function
import torch
import torchvision.models as models
import networks


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
        if source_size > 0:
            feats_source = feats[:source_size]
            logits = self.classifier(feats_source)
        else:
            logits = None
        grad_reversed_out = GradReverse.apply(feats, alpha)
        dom = self.domain_classifier(grad_reversed_out)
        return feats, logits, dom.squeeze()


class MNIST50Network(Network):
    """Network for the MNIST50 -> SVHN-B experiments"""
    
    def __init__(self):
        super(MNIST50Network, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AvgPool2d(kernel_size=6)
        )
        
        self.classifier = nn.Linear(in_features=128, out_features=10)
        self.domain_classifier = nn.Sequential(nn.Linear(128, 512), nn.ReLU(),
                                               nn.Dropout(),
                                               nn.Linear(512, 512), nn.ReLU(),
                                               nn.Dropout(),
                                               nn.Linear(512, 1)
                                               )


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

    def __init__(self, init_args):
        super().__init__()
        self.args = init_args
        self.backbone_file = self.args['backbone']
        self.validate_backbone_file()
        if self.backbone_file == "":
            print("Initializing new backbone...")
            self.backbone = models.resnet18(pretrained=False)
            self.fc_size = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone_file == "imagenet":
            print("Loading ILSVRC-pretrained backbone...")
            self.backbone = models.resnet18(pretrained=True)
            self.fc_size = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            print("Loading backbone weights from {}...".format(self.backbone_file))
            self.backbone = models.resnet18(pretrained=False)
            self.fc_size = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.load_state_dict(torch.load(self.backbone_file))
            
        self.classifier = nn.Linear(self.fc_size, 12)
        self.domain_classifier = nn.Sequential(nn.Linear(self.fc_size, 1000), nn.ReLU(), nn.Linear(1000, 1))
        
    def validate_backbone_file(self):
        """Checks whether provided backbone file name is valid"""
        import os
        if self.backbone_file == 'imagenet' or self.backbone_file == "" or os.path.isfile(self.backbone_file):
            return
        self.backbone_file = ""
        
        
def get_network(source_dataset, target_dataset, args):
    """
    Returns the preferred network for the specified datasets
    :return:
    """
    if source_dataset == 'svhn' or target_dataset == 'svhn':
        return SVHNNetwork()
    elif source_dataset == 'mnist50' and target_dataset == 'svhn-b':
        return networks.MNIST50DANNNetwork()
    elif 'mnist' in source_dataset and 'mnist' in target_dataset:
        return MNISTNetwork()
    elif 'toybox' in source_dataset or 'toybox' in target_dataset:
        return networks.ToyboxDANNNetwork(init_args=args)
    raise NotImplementedError("Network for src: {} and trgt: {} not implemented...".format(source_dataset,
                                                                                           target_dataset))
