"""
This module implements the Domain Adaptation by Backpropagation paper by Ganin and Lampitsky
"""
import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as torchdata
import torch.nn as nn
from torch.autograd import Function
import tqdm
import torch.utils.tensorboard as tb

import utils


class GradReverse(Function):
    """
    Gradient Reversal method
    """
    
    @staticmethod
    def forward(ctx, i):
        """forward method"""
        return i.view_as(i)
    
    @staticmethod
    def backward(ctx, grad_output):
        """backward method"""
        return grad_output * -1


def grad_reverse(x):
    """sdf"""
    return GradReverse.apply(x)


class Network(nn.Module):
    """
    Class definition for the domain adaptation module
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(num_features=64),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(num_features=64),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4)),
                                      nn.BatchNorm2d(num_features=128)
                                      )
        self.classifier = nn.Sequential(nn.Linear(128, 3072), nn.ReLU(), nn.BatchNorm1d(num_features=3072),
                                        nn.Linear(3072, 2048), nn.ReLU(), nn.BatchNorm1d(num_features=2048),
                                        nn.Linear(2048, 10)
                                        )
        self.domain_classifier = nn.Sequential(nn.Linear(128, 1024), nn.ReLU(), nn.BatchNorm1d(num_features=1024),
                                               nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(num_features=1024),
                                               nn.Linear(1024, 1)
                                               )
    
    def forward(self, x, source_size):
        """
        Forward method
        """
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        feats_source = feats[:source_size]
        logits = self.classifier(feats_source)
        grad_reversed_out = grad_reverse(feats)
        dom = self.domain_classifier(grad_reversed_out)
        return feats, logits, dom.squeeze()


class Experiment:
    """
    Class used to run the experiments
    """
    
    def __init__(self, train_on_svhn):
        self.train_on_svhn = train_on_svhn
        self.net = Network()
        
        self.optimizer = torch.optim.SGD(self.net.backbone.parameters(), lr=0.01, weight_decay=1e-5)
        self.optimizer.add_param_group({'params': self.net.classifier.parameters()})
        self.optimizer.add_param_group({'params': self.net.domain_classifier.parameters()})
        mnist_transform = transforms.Compose([transforms.Grayscale(3),
                                              transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0.5,
                                                                     brightness=0.3),
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1307, 0.1307, 0.1307),
                                                                   std=(0.3081, 0.3081, 0.3081))
                                              ])
        
        svhn_transform = transforms.Compose(
            [transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0.5, brightness=0.3),
             transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.4377, 0.4437, 0.4728),
                                  std=(0.1980, 0.2010, 0.1970))
             ])
        
        self.dataset1 = datasets.MNIST(root="../data/", train=True, transform=mnist_transform, download=True)
        self.dataset2 = datasets.SVHN(root="../data/", split='train', download=True, transform=svhn_transform)
        
        self.loader_1 = torchdata.DataLoader(self.dataset1, batch_size=128, shuffle=True, num_workers=4)
        self.loader_2 = torchdata.DataLoader(self.dataset2, batch_size=128, shuffle=True, num_workers=4)
        if self.train_on_svhn:
            self.loader_1, self.loader_2 = self.loader_2, self.loader_1
        self.test_dataset1 = datasets.MNIST(root="../data/", train=False, transform=mnist_transform, download=True)
        self.test_dataset2 = datasets.SVHN(root="../data/", split='test', download=True, transform=svhn_transform)
        
        self.test_loader_1 = torchdata.DataLoader(self.test_dataset1, batch_size=128, shuffle=True, num_workers=4)
        self.test_loader_2 = torchdata.DataLoader(self.test_dataset2, batch_size=128, shuffle=True, num_workers=4)
        
        self.net = self.net.cuda()
    
    def train(self):
        """
        Train network
        """
        import datetime
        dt_now = datetime.datetime.now()
        tb_writer = tb.SummaryWriter(log_dir="../runs/dann_temp_" + dt_now.strftime("%b-%d-%Y-%H-%M") + "/")
        loader_2_iter = iter(self.loader_2)
        num_epochs = 50
        self.net.backbone.train()
        self.net.classifier.train()
        self.net.domain_classifier.train()
        total_batches = 0
        for ep in range(1, num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.loader_1)
            for img1, labels1 in tqdm_bar:
                try:
                    img2, labels2 = next(loader_2_iter)
                except StopIteration:
                    loader_2_iter = iter(self.loader_2)
                    img2, labels2 = next(loader_2_iter)
                self.optimizer.zero_grad()
                
                images = torch.concat([img1, img2], dim=0)
                dom_labels = torch.cat([torch.zeros(img1.size(0)), torch.ones(img2.size(0))])
                images, dom_labels, labels1 = images.cuda(), dom_labels.cuda(), labels1.cuda()
                
                f, l, d = self.net.forward(images, img1.size(0))
                ce_loss = nn.CrossEntropyLoss()(l, labels1)
                total_batches += 1
                p = total_batches / (len(self.loader_1) * num_epochs)
                dom_pred = torch.sigmoid(d)
                dom_loss = nn.BCELoss()(dom_pred, dom_labels)
                import math
                lmbda = 2 / (1 + math.exp(-10 * p)) - 1
                total_loss = ce_loss + lmbda * dom_loss
                # dom_loss = torch.tensor([0.0])
                total_loss.backward()
                tqdm_bar.set_description("Ep: {}/{}  LR: {:.4f}  CE Loss: {:.4f}  Dom Loss: {:.4f} Lmbda: {:.4f}  "
                                         "Tot Loss: {:.4f}".format(ep, num_epochs, self.optimizer.param_groups[0]['lr'],
                                                                   ce_loss.item(), dom_loss.item(), lmbda,
                                                                   total_loss.item()))
                self.optimizer.step()
                tb_writer.add_scalar(tag="LR", scalar_value=self.optimizer.param_groups[0]['lr'],
                                     global_step=total_batches)
                tb_writer.add_scalar(tag="Lambda", scalar_value=lmbda, global_step=total_batches)
                tb_writer.add_scalar(tag="p", scalar_value=p, global_step=total_batches)
                tb_writer.add_scalars(main_tag="Training", tag_scalar_dict={'CE Loss': ce_loss.item(),
                                                                            'Dom Loss': dom_loss.item(),
                                                                            'Total Loss': total_loss.item()
                                                                            },
                                      global_step=total_batches)
                
            if (ep + 1) % 10 == 0:
                self.optimizer.param_groups[0]['lr'] *= 0.9
        tb_writer.close()
        
    def eval(self):
        """
        Eval network on test datasets
        """
        self.net.backbone.eval()
        self.net.classifier.eval()
        self.net.domain_classifier.eval()
        dataloaders = [self.test_loader_1, self.test_loader_2]
        d_names = ['mnist', 'svhn']
        for idx in range(2):
            loader = dataloaders[idx]
            total_num = 0
            correct_num = 0
            for images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    _, logits, _ = self.net.forward(images, images.size(0))
                    res, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                    total_num += logits.size(0)
                    correct_num += res[0].item() * logits.size(0)
            acc = correct_num / total_num
            print("Accuracy on dataset {}: {:.2f}".format(d_names[idx], acc))


if __name__ == "__main__":
    exp = Experiment(train_on_svhn=True)
    exp.train()
    exp.eval()
