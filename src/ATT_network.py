"""Network for ATT algorithm"""
import torch.nn as nn
import torch


class View(nn.Module):
    """Module for reshaping input"""
    def __init__(self):
        super(View, self).__init__()
    
    @staticmethod
    def forward(x):
        """forward method"""
        return x.view(x.size(0), -1)
    
    
class Network(nn.Module):
    """
    Class definition for the domain adaptation network module
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.classifier_source_1 = nn.Identity()
        self.classifier_source_2 = nn.Identity()
        self.classifier_target = nn.Identity()
    
    def forward(self, x, source_size):
        """
        Forward method
        """
        feats = self.backbone.forward(x)
        target_feats = feats[source_size:]
        source_logits_1 = self.classifier_source_1.forward(feats)
        source_logits_2 = self.classifier_source_2.forward(feats)
        target_logits = self.classifier_target.forward(target_feats)
        return source_logits_1, source_logits_2, target_logits


class SVHNNetwork(Network):
    """
    Network for experiments with SVHN
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding='same'),
                                      nn.ReLU(), nn.Dropout(),
                                      View(),
                                      nn.Linear(8192, 3072), nn.ReLU(), nn.Dropout()
                                      )
        
        self.classifier_source_1 = nn.Sequential()
        self.classifier_source_1.add_module("fc_1", nn.Linear(3072, 2048))
        self.classifier_source_1.add_module("bn", nn.BatchNorm1d(num_features=2048))
        self.classifier_source_1.add_module("relu", nn.ReLU())
        self.classifier_source_1.add_module("dropout", nn.Dropout())
        self.classifier_source_1.add_module("fc_2", nn.Linear(2048, 10))

        self.classifier_source_2 = nn.Sequential()
        self.classifier_source_2.add_module("fc_1", nn.Linear(3072, 2048))
        self.classifier_source_2.add_module("bn", nn.BatchNorm1d(num_features=2048))
        self.classifier_source_2.add_module("relu", nn.ReLU())
        self.classifier_source_2.add_module("dropout", nn.Dropout())
        self.classifier_source_2.add_module("fc_2", nn.Linear(2048, 10))

        self.classifier_target = nn.Sequential()
        self.classifier_target.add_module("fc_1", nn.Linear(3072, 2048))
        # self.classifier_target.add_module("bn", nn.BatchNorm1d(num_features=2048))
        self.classifier_target.add_module("relu", nn.ReLU())
        self.classifier_target.add_module("dropout", nn.Dropout())
        self.classifier_target.add_module("fc_2", nn.Linear(2048, 10))
        
        
def main():
    """test networks"""
    n = SVHNNetwork()
    n.forward(x=None, source_size=None)
    

if __name__ == "__main__":
    main()
