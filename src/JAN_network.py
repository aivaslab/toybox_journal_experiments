"""
Networks for JAN experiments
"""
import torch
import torch.nn as nn
import torchvision.models as models
import os


class JANNet(torch.nn.Module):
    """Network for JAN experiments"""
    
    def __init__(self, backbone_file, num_classes):
        super(JANNet, self).__init__()
        self.backbone_file = backbone_file
        self.validate_backbone_file()
        self.num_classes = num_classes
        if self.backbone_file == "imagenet":
            print("Loading backbone weights from ILSVRC trained model...")
            self.backbone = models.resnet18(pretrained=True)
        else:
            print("Initializing new backbone...")
            self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, self.num_classes)
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
    

def get_network(args):
    """Returns the specified model for JAN experiments"""
    if 'toybox' in args['datasets']:
        return JANNet(backbone_file=args['backbone'], num_classes=12)
    else:
        return JANNet(backbone_file=args['backbone'], num_classes=31)
