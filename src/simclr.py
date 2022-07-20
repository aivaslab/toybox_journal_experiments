"""
Module to implement the simclr algorithm
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torchvision.models as models
import tqdm
import torch.utils.tensorboard as tb
import argparse
import os
import torch.utils.data as torchdata

import dataset_ssl_in12

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)


def info_nce_loss(features, temp):
    """
    loss function for SimCLR
    """
    dev = torch.device('cuda:0')
    batch_size = features.shape[0] / 2
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dev)
    features = f.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(dev)
    labels = labels[~mask].view(labels.shape[0], -1).type(torch.uint8)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dev)
    logits = logits / temp
    return logits, labels


class Network(nn.Module):
    """
    SimCLR Network
    """
    
    def __init__(self, backbone_file=""):
        super().__init__()
        self.backbone_file = backbone_file
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if os.path.isfile(self.backbone_file):
            print("Loading weights from:", self.backbone_file)
            self.backbone.load_state_dict(torch.load(self.backbone_file))
        self.fc = nn.Sequential(nn.Linear(self.fc_size, self.fc_size), nn.BatchNorm1d(num_features=self.fc_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.fc_size, 128), nn.BatchNorm1d(num_features=128))
    
    def forward(self, x):
        """
        forward method for network
        """
        y = self.backbone(x)
        z = self.fc(y)
        return z


class SimCLR:
    """
    Class implementing SimCLR algorithm
    """

    def __init__(self, exp_args):
        self.args = exp_args
        self.num_epochs = self.args['num_epochs']
        self.lr = self.args['lr']
        self.save = self.args['save']
        self.b_size = self.args['batch_size']
        self.fraction = self.args['fraction']
        self.backbone_file_name = self.args['backbone_file']
        self.layers_frozen = self.args['layers_frozen']
        self.save_dir = self.args['save_dir']
        self.net = Network(backbone_file=self.backbone_file_name)

        color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.2, saturation=0.8)
        gaussian_blur = transforms.GaussianBlur(kernel_size=21, sigma=1.0)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([gaussian_blur], p=0.5),
            transforms.Resize(270),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
        ])
        
        self.train_data = dataset_ssl_in12.DatasetIN12(fraction=self.fraction, transform=self.train_transform)
        self.train_loader = torchdata.DataLoader(self.train_data, batch_size=self.b_size, shuffle=True,
                                                 num_workers=6)
        
    def prepare_network_for_training(self):
        """
        Adjusts the requires_grad parameters of the network and initializes optimizer
        """
        for params in self.net.fc.parameters():
            params.requires_grad = True
        for params in self.net.backbone.parameters():
            params.requires_grad = True
        self.net.fc.train()
        optimizer = torch.optim.Adam(self.net.fc.parameters(), lr=self.lr, weight_decay=1e-6)
    
        if self.layers_frozen > -1:
            for params in self.net.backbone.conv1.parameters():
                params.requires_grad = False
            for params in self.net.backbone.bn1.parameters():
                params.requires_grad = False
            for params in self.net.backbone.relu.parameters():
                params.requires_grad = False
            for params in self.net.backbone.maxpool.parameters():
                params.requires_grad = False
        else:
            optimizer.add_param_group({'params': self.net.backbone.conv1.parameters()})
            optimizer.add_param_group({'params': self.net.backbone.bn1.parameters()})
            optimizer.add_param_group({'params': self.net.backbone.relu.parameters()})
            optimizer.add_param_group({'params': self.net.backbone.maxpool.parameters()})
        layer_mode = False if self.layers_frozen > -1 else True
        self.net.backbone.conv1.train(mode=layer_mode)
        self.net.backbone.bn1.train(mode=layer_mode)
        self.net.backbone.relu.train(mode=layer_mode)
        self.net.backbone.maxpool.train(mode=layer_mode)
    
        if self.layers_frozen > 0:
            for params in self.net.backbone.layer1.parameters():
                params.requires_grad = False
        else:
            optimizer.add_param_group({'params': self.net.backbone.layer1.parameters()})
        layer_mode = False if self.layers_frozen > 0 else True
        self.net.backbone.layer1.train(mode=layer_mode)
    
        if self.layers_frozen > 1:
            for params in self.net.backbone.layer2.parameters():
                params.requires_grad = False
        else:
            optimizer.add_param_group({'params': self.net.backbone.layer2.parameters()})
        layer_mode = False if self.layers_frozen > 1 else True
        self.net.backbone.layer2.train(mode=layer_mode)
    
        if self.layers_frozen > 2:
            for params in self.net.backbone.layer3.parameters():
                params.requires_grad = False
        else:
            optimizer.add_param_group({'params': self.net.backbone.layer3.parameters()})
        layer_mode = False if self.layers_frozen > 2 else True
        self.net.backbone.layer3.train(mode=layer_mode)
    
        if self.layers_frozen > 3:
            for params in self.net.backbone.layer4.parameters():
                params.requires_grad = False
            for params in self.net.backbone.avgpool.parameters():
                params.requires_grad = False
        else:
            optimizer.add_param_group({'params': self.net.backbone.layer4.parameters()})
            optimizer.add_param_group({'params': self.net.backbone.avgpool.parameters()})
        layer_mode = False if self.layers_frozen > 3 else True
        self.net.backbone.layer4.train(mode=layer_mode)
        self.net.backbone.avgpool.train(mode=layer_mode)
        total_params_backbone = sum(p.numel() for p in self.net.backbone.parameters())
        train_params_backbone = sum(p.numel() for p in self.net.backbone.parameters() if p.requires_grad)
    
        total_params_fc = sum(p.numel() for p in self.net.fc.parameters())
        train_params_fc = sum(p.numel() for p in self.net.fc.parameters() if p.requires_grad)
        print("{}/{} parameters in backbone are trainable.".format(train_params_backbone, total_params_backbone))
        print("{}/{} parameters in projection layer are trainable.".format(train_params_fc, total_params_fc))
    
        self.net.cuda()
    
        return optimizer
        
    def train(self):
        """
        Train the network using the SimCLR algorithm
        """
        if self.save:
            tb_writer = tb.SummaryWriter(log_dir="../runs/" + self.save_dir + "/")
        optimizer = self.prepare_network_for_training()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=self.num_epochs * len(self.train_loader),
                                                                  eta_min=1e-6)
        if self.num_epochs == 0:
            return
        total_batches = 0
        for ep in range(1, self.num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.train_loader)
            batches = 0
            total_loss = 0.0
            for idx, images in tqdm_bar:
                optimizer.zero_grad()
                images = torch.cat(images, 0)
                images = images.cuda()
                feats = self.net.forward(images)
                logits, labels = info_nce_loss(feats, temp=0.1)
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()
                batches += 1
                total_batches += 1
                tqdm_bar.set_description("Epochs: {}/{}  LR: {:.4f}  Loss:{:.6f}"
                                         .format(ep, self.num_epochs, optimizer.param_groups[0]['lr'],
                                                 total_loss/batches))
                if self.save:
                    tb_writer.add_scalar(tag='Loss/Batch', scalar_value=loss.item(), global_step=total_batches)
                    tb_writer.add_scalar(tag='Loss/Avg', scalar_value=total_loss/batches,
                                         global_step=total_batches)
                    tb_writer.add_scalar(tag='LR', scalar_value=optimizer.param_groups[0]['lr'],
                                         global_step=total_batches)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            tqdm_bar.close()
        import os
        if not os.path.isdir("../out/temp/"):
            os.mkdir("../out/temp/")
        torch.save(self.net.backbone.state_dict(), "../out/" + self.save_dir + "/ssl_resnet18_backbone.pt")
        dummy_classifier = nn.Linear(512, 12)
        torch.save(dummy_classifier.state_dict(), "../out/" + self.save_dir + "/ssl_resnet18_classifier.pt")
        if self.save:
            tb_writer.close()


def get_parser():
    """
    Create parser and return it for the algorithm
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_epochs", "-e", default=10, type=int)
    parser.add_argument("--save", "-s", default=False, action='store_true')
    parser.add_argument("--lr", "-lr", type=float, default=1e-1)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--fraction", "-f", type=float, default=0.1)
    parser.add_argument("--backbone_file", "-bf", type=str, default="")
    parser.add_argument("--layers_frozen", "-lf", type=int, default=-1)
    parser.add_argument("--save_dir", "-sd", default="temp", type=str)
    return parser.parse_args()

    
if __name__ == "__main__":
    args = vars(get_parser())
    exp = SimCLR(exp_args=args)
    exp.train()
