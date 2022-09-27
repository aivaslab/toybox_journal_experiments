"""
Network for experiments with mnist and svhn
"""
import torch.nn as nn
import torch.nn.functional as functional


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
        x = functional.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = functional.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(functional.relu(self.conv1_3_bn(self.conv1_3(x))))
        x = self.drop1(x)
        
        x = functional.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = functional.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(functional.relu(self.conv2_3_bn(self.conv2_3(x))))
        x = self.drop2(x)
        
        x = functional.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = functional.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = functional.relu(self.nin3_3_bn(self.nin3_3(x)))
        
        x = functional.avg_pool2d(x, 6)
        x = x.view(-1, 128)
        
        x = self.fc4(x)
        return x
    
    def get_params(self, frozen_layers):
        """
        Returns the parameters from the un-frozen layers
        """
        params = nn.ParameterList()
        params.extend(self.fc4.parameters())
        if frozen_layers < 3:
            params.extend(self.conv3_1.parameters())
            params.extend(self.conv3_1_bn.parameters())
            params.extend(self.nin3_2.parameters())
            params.extend(self.nin3_2_bn.parameters())
            params.extend(self.nin3_3.parameters())
            params.extend(self.nin3_3_bn.parameters())
        
        if frozen_layers < 2:
            params.extend(self.conv2_1.parameters())
            params.extend(self.conv2_1_bn.parameters())
            params.extend(self.conv2_2.parameters())
            params.extend(self.conv2_2_bn.parameters())
            params.extend(self.conv2_3.parameters())
            params.extend(self.conv2_3_bn.parameters())
            params.extend(self.pool2.parameters())
            params.extend(self.drop2.parameters())
        
        if frozen_layers < 1:
            params.extend(self.conv1_1.parameters())
            params.extend(self.conv1_1_bn.parameters())
            params.extend(self.conv1_2.parameters())
            params.extend(self.conv1_2_bn.parameters())
            params.extend(self.conv1_3.parameters())
            params.extend(self.conv1_3_bn.parameters())
            params.extend(self.pool1.parameters())
            params.extend(self.drop1.parameters())
        return params
    
    def set_train_mode(self):
        """
        Set all params in training mode
        """
        self.conv1_1.train()
        self.conv1_1_bn.train()
        self.conv1_2.train()
        self.conv1_2_bn.train()
        self.conv1_3.train()
        self.conv1_3_bn.train()
        self.pool1.train()
        self.drop1.train()
        
        self.conv2_1.train()
        self.conv2_1_bn.train()
        self.conv2_2.train()
        self.conv2_2_bn.train()
        self.conv2_3.train()
        self.conv2_3_bn.train()
        self.pool2.train()
        self.drop2.train()
        
        self.conv3_1.train()
        self.conv3_1_bn.train()
        self.nin3_2.train()
        self.nin3_2_bn.train()
        self.nin3_3.train()
        self.nin3_3_bn.train()
        
        self.fc4.train()
    
    def set_eval_mode(self):
        """
        Set all params in eval mode
        """
        self.conv1_1.eval()
        self.conv1_1_bn.eval()
        self.conv1_2.eval()
        self.conv1_2_bn.eval()
        self.conv1_3.eval()
        self.conv1_3_bn.eval()
        self.pool1.eval()
        self.drop1.eval()
        
        self.conv2_1.eval()
        self.conv2_1_bn.eval()
        self.conv2_2.eval()
        self.conv2_2_bn.eval()
        self.conv2_3.eval()
        self.conv2_3_bn.eval()
        self.pool2.eval()
        self.drop2.eval()
        
        self.conv3_1.eval()
        self.conv3_1_bn.eval()
        self.nin3_2.eval()
        self.nin3_2_bn.eval()
        self.nin3_3.eval()
        self.nin3_3_bn.eval()
        
        self.fc4.eval()
        

class Network2(nn.Module):
    """
    network for mnist -> svhn
    """
    def __init__(self, n_classes):
        super(Network2, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 48, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(96)
        self.conv2_2 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(96)
        self.conv2_3 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(96, 192, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(192)
        self.conv3_3 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(192)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(192, 192)
        self.fc5 = nn.Linear(192, n_classes)

    def forward(self, x):
        """Forward method"""
        x = functional.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = self.pool1(functional.relu(self.conv1_2_bn(self.conv1_2(x))))

        x = functional.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = functional.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(functional.relu(self.conv2_3_bn(self.conv2_3(x))))

        x = functional.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = functional.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = self.pool3(functional.relu(self.conv3_3_bn(self.conv3_3(x))))

        x = functional.avg_pool2d(x, 4)
        x = x.view(-1, 192)
        x = self.drop1(x)

        x = functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def set_train_mode(self):
        """
        Set all params in training mode
        """
        self.conv1_1.train()
        self.conv1_1_bn.train()
        self.conv1_2.train()
        self.conv1_2_bn.train()
        self.pool1.train()
    
        self.conv2_1.train()
        self.conv2_1_bn.train()
        self.conv2_2.train()
        self.conv2_2_bn.train()
        self.conv2_3.train()
        self.conv2_3_bn.train()
        self.pool2.train()
    
        self.conv3_1.train()
        self.conv3_1_bn.train()
        self.conv3_2.train()
        self.conv3_2_bn.train()
        self.conv3_3.train()
        self.conv3_3_bn.train()
        self.pool3.train()
        self.drop11.train()
    
        self.fc4.train()
        self.fc5.train()

    def set_eval_mode(self):
        """
        Set all params in eval mode
        """
        self.conv1_1.eval()
        self.conv1_1_bn.eval()
        self.conv1_2.eval()
        self.conv1_2_bn.eval()
        self.pool1.eval()

        self.conv2_1.eval()
        self.conv2_1_bn.eval()
        self.conv2_2.eval()
        self.conv2_2_bn.eval()
        self.conv2_3.eval()
        self.conv2_3_bn.eval()
        self.pool2.eval()

        self.conv3_1.eval()
        self.conv3_1_bn.eval()
        self.conv3_2.eval()
        self.conv3_2_bn.eval()
        self.conv3_3.eval()
        self.conv3_3_bn.eval()
        self.pool3.eval()
        self.drop11.eval()

        self.fc4.eval()
        self.fc5.eval()

    def get_params(self, frozen_layers):
        """
        Returns the parameters from the un-frozen layers
        """
        params = nn.ParameterList()
        params.extend(self.fc4.parameters())
        params.extend(self.fc5.parameters())
        if frozen_layers < 3:
            params.extend(self.conv3_1.parameters())
            params.extend(self.conv3_1_bn.parameters())
            params.extend(self.conv3_2.parameters())
            params.extend(self.conv3_2_bn.parameters())
            params.extend(self.conv3_3.parameters())
            params.extend(self.conv3_3_bn.parameters())
            params.extend(self.pool3.parameters())
            params.extend(self.drop1.parameters())
    
        if frozen_layers < 2:
            params.extend(self.conv2_1.parameters())
            params.extend(self.conv2_1_bn.parameters())
            params.extend(self.conv2_2.parameters())
            params.extend(self.conv2_2_bn.parameters())
            params.extend(self.conv2_3.parameters())
            params.extend(self.conv2_3_bn.parameters())
            params.extend(self.pool2.parameters())
    
        if frozen_layers < 1:
            params.extend(self.conv1_1.parameters())
            params.extend(self.conv1_1_bn.parameters())
            params.extend(self.conv1_2.parameters())
            params.extend(self.conv1_2_bn.parameters())
            params.extend(self.pool1.parameters())
        return params
    

if __name__ == "__main__":
    net = Network(n_classes=10)
    all_params = net.get_params(3)
    print(all_params)
    import torch
    optimer = torch.optim.SGD(all_params, lr=0.01)
    