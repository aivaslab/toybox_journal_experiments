"""
Module to implement the simclr algorithm
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torchvision.models as models
import tqdm

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
    
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(self.fc_size, self.fc_size), nn.ReLU(inplace=True),
                                nn.Linear(self.fc_size, 128))
    
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
    
    def __init__(self):
        self.net = Network()
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
        ])
        self.train_data = dataset_ssl_in12.DatasetIN12(fraction=0.5, transform=self.train_transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True, num_workers=4)
    
    def train(self, num_epochs):
        """
        Train the network using the SimCLR algorithm
        """
        for params in self.net.backbone.parameters():
            params.requires_grad = True
        for params in self.net.fc.parameters():
            params.requires_grad = True
        self.net.backbone.train()
        self.net.fc.train()
        self.net.cuda()
        optimizer = torch.optim.Adam(self.net.backbone.parameters(), lr=1e-1, weight_decay=1e-6)
        optimizer.add_param_group({'params': self.net.fc.parameters()})
        
        for ep in range(1, num_epochs + 1):
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
                tqdm_bar.set_description("Epochs: {}/{}  LR: {:.4f}  Loss:{:.6f}"
                                         .format(ep, num_epochs, optimizer.param_groups[0]['lr'], total_loss/batches))
                loss.backward()
                optimizer.step()
            if (ep + 1) % 3 == 0:
                optimizer.param_groups[0]['lr'] *= 0.8
            tqdm_bar.close()
        import os
        os.mkdir("../runs/temp/")
        torch.save(self.net.backbone.state_dict(), "../out/temp/ssl_resnet18_backbone.pt")
        dummy_classifier = nn.Linear(512, 12)
        torch.save(dummy_classifier.state_dict(), "../out/temp/ssl_resnet18_classifier.pt")


if __name__ == "__main__":
    exp = SimCLR()
    exp.train(num_epochs=5)
