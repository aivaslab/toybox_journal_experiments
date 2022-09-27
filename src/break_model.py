"""sdf"""
import torch
import os
import torch.nn as nn
import torchvision.models as models
import utils


class NetworkMeanTeacher(nn.Module):
    """ResNet-18 backbone for toybox and in12 training"""
    
    def __init__(self, pretrained):
        super(NetworkMeanTeacher, self).__init__()
        self.pretrained = pretrained
        self.backbone = models.resnet18(pretrained=self.pretrained)
        self.backbone.apply(utils.weights_init)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
        self.classifier.apply(utils.weights_init)
    
    def forward(self, x):
        """forward for module"""
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
    
    def set_train_mode(self):
        """set network in train mode"""
        self.backbone.train()
        self.classifier.train()
    
    def set_eval_mode(self):
        """set network in eval mode"""
        self.backbone.eval()
        self.classifier.eval()


def break_mod(dir_name):
    """ds"""
    f_name = dir_name + "teacher_final.pt"
    assert(os.path.isfile(f_name))
    net = NetworkMeanTeacher(pretrained=False)
    net.load_state_dict(torch.load(f_name))
    backbone_file_name = dir_name + "mt_resnet18_backbone.pt"
    classifier_file_name = dir_name + "mt_resnet18_classifier.pt"
    torch.save(net.backbone.state_dict(), backbone_file_name)
    torch.save(net.classifier.state_dict(), classifier_file_name)


if __name__ == "__main__":
    break_mod(dir_name="../out/TB_IN12_SELF_ENSEMBLE/Sep-15-2022-23-14/")
