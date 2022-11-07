"""Networks for JAN, DANN, SE algorithms"""
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Function
import os


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


class MNIST50Backbone(nn.Module):
    """Module defining the backbone for MNIST50 experiments"""
    
    def __init__(self):
        super(MNIST50Backbone, self).__init__()
        self.network = nn.Sequential()
        self.network.add_module('conv_1_1', nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=1))
        self.network.add_module('bn_1_1', nn.BatchNorm2d(128))
        self.network.add_module('relu_1_1', nn.ReLU())
        self.network.add_module('conv_1_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                                      padding=1))
        self.network.add_module('bn_1_2', nn.BatchNorm2d(128))
        self.network.add_module('relu_1_2', nn.ReLU())
        self.network.add_module('conv_1_3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                                      padding=1))
        self.network.add_module('bn_1_3', nn.BatchNorm2d(128))
        self.network.add_module('relu_1_3', nn.ReLU())
        
        self.network.add_module('max_pool_1', nn.MaxPool2d(kernel_size=(2, 2)))
        self.network.add_module('drop_1', nn.Dropout())
        
        self.network.add_module('conv_2_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                                      padding=1))
        self.network.add_module('bn_2_1', nn.BatchNorm2d(256))
        self.network.add_module('relu_2_1', nn.ReLU())
        self.network.add_module('conv_2_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                      padding=1))
        self.network.add_module('bn_2_2', nn.BatchNorm2d(256))
        self.network.add_module('relu_2_2', nn.ReLU())
        self.network.add_module('conv_2_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                      padding=1))
        self.network.add_module('bn_2_3', nn.BatchNorm2d(256))
        self.network.add_module('relu_2_3', nn.ReLU())
        self.network.add_module('max_pool_2', nn.MaxPool2d(kernel_size=(2, 2)))
        self.network.add_module('drop_2', nn.Dropout())
        
        self.network.add_module('conv_3_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                                      padding=0))
        self.network.add_module('bn_3_1', nn.BatchNorm2d(512))
        self.network.add_module('relu_3_1', nn.ReLU())
        self.network.add_module('conv_3_2', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1),
                                                      padding=1))
        self.network.add_module('bn_3_2', nn.BatchNorm2d(256))
        self.network.add_module('relu_3_2', nn.ReLU())
        self.network.add_module('conv_3_3', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1),
                                                      padding=1))
        self.network.add_module('bn_3_3', nn.BatchNorm2d(128))
        self.network.add_module('relu_3_3', nn.ReLU())
        self.network.add_module('avgpool', nn.AvgPool2d(kernel_size=6))
        
    def forward(self, x):
        """forward method"""
        return self.network(x)
        
        
class DANNBaseNetwork(nn.Module):
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


class MNIST50DANNNetwork(DANNBaseNetwork):
    """Network for the MNIST50 -> SVHN-B experiments"""
    
    def __init__(self):
        super(MNIST50DANNNetwork, self).__init__()
        self.backbone = MNIST50Backbone().network
        
        self.classifier = nn.Linear(in_features=128, out_features=10)
        self.domain_classifier = nn.Sequential(nn.Linear(128, 512), nn.ReLU(),
                                               nn.Dropout(),
                                               nn.Linear(512, 512), nn.ReLU(),
                                               nn.Dropout(),
                                               nn.Linear(512, 1)
                                               )


class ToyboxDANNNetwork(DANNBaseNetwork):
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
        if self.backbone_file == 'imagenet' or self.backbone_file == "" or os.path.isfile(self.backbone_file):
            return
        self.backbone_file = ""


class MNIST50JANNetwork(nn.Module):
    """Network for the MNIST50 -> SVHN-B experiments"""
    
    def __init__(self):
        super(MNIST50JANNetwork, self).__init__()
        self.backbone = MNIST50Backbone().network
        
        self.classifier = nn.Linear(in_features=128, out_features=10)
        self.backbone_file = ""
    
    def forward(self, x):
        """
        Forward method for network
        """
        feats = self.backbone(x)
        feats_view = feats.view(feats.size(0), -1)
        logits = self.classifier(feats_view)
        return feats.squeeze(), logits
    
    def set_train_mode(self):
        """
        Set all params in training mode
        """
        self.backbone.train()
        self.classifier.train()

    def set_eval_mode(self):
        """
        Set all params in training mode
        """
        self.backbone.eval()
        self.classifier.eval()


class ToyboxJANNetwork(nn.Module):
    """Network for JAN experiments"""
    
    def __init__(self, backbone_file):
        super(ToyboxJANNetwork, self).__init__()
        self.backbone_file = backbone_file
        self.validate_backbone_file()
        if self.backbone_file == "imagenet":
            print("Loading backbone weights from ILSVRC trained model...")
            self.backbone = models.resnet18(pretrained=True)
        else:
            print("Initializing new backbone...")
            self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
        if self.backbone_file != "" and self.backbone_file != "imagenet":
            print("Loading backbone weights from {}...".format(self.backbone_file))
            self.backbone.load_state_dict(torch.load(self.backbone_file))
    
    def forward(self, x):
        """
        Forward method for network
        """
        feats = self.backbone(x)
        feats_view = feats.view(feats.size(0), -1)
        logits = self.classifier(feats_view)
        return feats, logits
    
    def validate_backbone_file(self):
        """Validate backbone file"""
        if self.backbone_file != "" and self.backbone_file != "imagenet" and not os.path.isfile(self.backbone_file):
            self.backbone_file = ""
        return
    
    def set_train_mode(self):
        """
        Set all params in training mode
        """
        self.backbone.train()
        self.classifier.train()

    def set_eval_mode(self):
        """
        Set all params in training mode
        """
        self.backbone.eval()
        self.classifier.eval()
    

def get_network_dann(args):
    """Returns the specified model for JAN experiments"""
    if 'toybox' in args['datasets']:
        return ToyboxDANNNetwork(init_args=args)
    elif 'mnist50' in args['datasets']:
        return MNIST50DANNNetwork()
    else:
        raise NotImplementedError("Network not implemented for source dataset {}...".format(args['datasets']))
    
    
def get_network(args):
    """Returns the specified model for JAN experiments"""
    if 'toybox' in args['datasets']:
        return ToyboxJANNetwork(backbone_file=args['backbone'])
    elif 'mnist50' in args['datasets']:
        return MNIST50JANNetwork()
    else:
        raise NotImplementedError("Network not implemented for source dataset {}...".format(args['datasets']))


def main():
    """Main method"""
    model = MNIST50DANNNetwork()
    print(sum([p.numel() for p in model.backbone.parameters()]))
    
    
if __name__ == "__main__":
    main()
    