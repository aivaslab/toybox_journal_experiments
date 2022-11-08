"""Module to implement the Linear Evaluator"""
import os.path

import torchvision.models as models
import torch
import torch.nn as nn


class ResNet18WithActivations(nn.Module):
    """Extract activations from a ResNet-18 model during forward pass"""
    MODULES = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    
    MAX_POOLS = {
        'conv1': nn.MaxPool2d(kernel_size=14, stride=14, padding=0),
        'layer1': nn.MaxPool2d(kernel_size=7, stride=7, padding=0),
        'layer2': nn.MaxPool2d(kernel_size=5, stride=5, padding=2),
        'layer3': nn.MaxPool2d(kernel_size=4, stride=4, padding=1),
        'layer4': nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        'avgpool': nn.Identity()
    }
    
    FC_SIZES = {
        'conv1': 4096,
        'layer1': 4096,
        'layer2': 4608,
        'layer3': 4096,
        'layer4': 4608,
        'avgpool': 512
    }
    
    def __init__(self, backbone_file):
        super().__init__()
        self.backbone_file = backbone_file
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        # self.backbone.load_state_dict(torch.load(self.backbone_file))
        
        for module in self.MODULES:
            module_ = self.backbone.__getattr__(module)
            module_.register_forward_hook(self.get_hook(module))
        
        self.activation = None
        self.output_dict = {}
        self.input_dict = {}
        
    def __str__(self):
        return "ResNet18 With Activations"
    
    def get_hook(self, name):
        """kn"""
        
        def act(model, inp, output):
            """ s"""
            self.output_dict[name] = output.squeeze().detach()
            self.input_dict[name] = inp[0].squeeze().detach()
        
        return act
    
    def forward(self, x):
        """forward method"""
        self.backbone.forward(x)
    
    def get_activation(self, name):
        """Returns activation after forward pass"""
        if name not in self.MODULES:
            raise NotImplementedError("Forward hook for layer {} not implemented...".format(name))
        activation = self.output_dict[name]
        reduced = self.MAX_POOLS[name](activation)
        print(activation.shape, reduced.shape)
        return reduced.view(activation.size(0), -1)


class MNIST50NetworkWithActivations(nn.Module):
    """Network for the MNIST50 -> SVHN-B experiments"""

    MODULES = ['drop_1', 'drop_2', 'relu_3_3', 'avgpool']

    MAX_POOLS = {
        'drop_1': nn.MaxPool2d(kernel_size=5, stride=5),
        'drop_2': nn.MaxPool2d(kernel_size=4, stride=4),
        'relu_3_3': nn.MaxPool2d(kernel_size=2, stride=2),
        'avgpool': nn.Identity()
    }

    FC_SIZES = {
        'drop_1': 1152,
        'drop_2': 1024,
        'relu_3_3': 1152,
        'avgpool': 128,
    }
    
    def __init__(self, backbone_file):
        super(MNIST50NetworkWithActivations, self).__init__()
        self.backbone_file = backbone_file
        self.backbone = nn.Sequential()
        self.backbone.add_module('conv_1_1', nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=1))
        self.backbone.add_module('bn_1_1', nn.BatchNorm2d(128))
        self.backbone.add_module('relu_1_1', nn.ReLU())
        self.backbone.add_module('conv_1_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                                       padding=1))
        self.backbone.add_module('bn_1_2', nn.BatchNorm2d(128))
        self.backbone.add_module('relu_1_2', nn.ReLU())
        self.backbone.add_module('conv_1_3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                                       padding=1))
        self.backbone.add_module('bn_1_3', nn.BatchNorm2d(128))
        self.backbone.add_module('relu_1_3', nn.ReLU())

        self.backbone.add_module('max_pool_1', nn.MaxPool2d(kernel_size=(2, 2)))
        self.backbone.add_module('drop_1', nn.Dropout())

        self.backbone.add_module('conv_2_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                                       padding=1))
        self.backbone.add_module('bn_2_1', nn.BatchNorm2d(256))
        self.backbone.add_module('relu_2_1', nn.ReLU())
        self.backbone.add_module('conv_2_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                       padding=1))
        self.backbone.add_module('bn_2_2', nn.BatchNorm2d(256))
        self.backbone.add_module('relu_2_2', nn.ReLU())
        self.backbone.add_module('conv_2_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                       padding=1))
        self.backbone.add_module('bn_2_3', nn.BatchNorm2d(256))
        self.backbone.add_module('relu_2_3', nn.ReLU())
        self.backbone.add_module('max_pool_2', nn.MaxPool2d(kernel_size=(2, 2)))
        self.backbone.add_module('drop_2', nn.Dropout())

        self.backbone.add_module('conv_3_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                                       padding=0))
        self.backbone.add_module('bn_3_1', nn.BatchNorm2d(512))
        self.backbone.add_module('relu_3_1', nn.ReLU())
        self.backbone.add_module('conv_3_2', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1),
                                                       padding=0))
        self.backbone.add_module('bn_3_2', nn.BatchNorm2d(256))
        self.backbone.add_module('relu_3_2', nn.ReLU())
        self.backbone.add_module('conv_3_3', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1),
                                                       padding=0))
        self.backbone.add_module('bn_3_3', nn.BatchNorm2d(128))
        self.backbone.add_module('relu_3_3', nn.ReLU())
        self.backbone.add_module('avgpool', nn.AvgPool2d(kernel_size=6))
        
        self.backbone.load_state_dict(torch.load(self.backbone_file))

        for module in self.MODULES:
            module_ = self.backbone.__getattr__(module)
            module_.register_forward_hook(self.get_hook(module))

        self.activation = None
        self.output_dict = {}
        self.input_dict = {}

    def __str__(self):
        return "MNIST50Network With Activations"

    def get_hook(self, name):
        """kn"""
    
        def act(model, inp, output):
            """ s"""
            self.output_dict[name] = output.squeeze().detach()
            self.input_dict[name] = inp[0].squeeze().detach()
    
        return act
    
    def get_activation(self, name):
        """Returns activation after forward pass"""
        if name not in self.MODULES:
            raise NotImplementedError("Forward hook for layer {} not implemented...".format(name))
        activation = self.output_dict[name]
        reduced = self.MAX_POOLS[name](activation)
        return reduced

    def forward(self, x):
        """Forward prop for the network"""
        self.backbone.forward(x)

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


def main():
    """Main method"""
    res = ResNet18WithActivations(backbone_file="")
    with torch.no_grad():
        inp = torch.rand(size=(3, 3, 224, 224))
        res.forward(inp)
    for mod in res.MODULES:
        activation = res.get_activation(mod)
        print(mod, activation.shape, activation.view(activation.size(0), -1).shape)
        layer = nn.Linear(res.FC_SIZES[mod], 12)
        activation = activation.view(activation.size(0), -1)
        print(layer.forward(activation).shape)


if __name__ == "__main__":
    main()
