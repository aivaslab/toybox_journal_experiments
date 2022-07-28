"""
Domain Adaptation between Toybox and IN-12
"""

import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as torchdata
import torch.nn as nn
from torch.autograd import Function
import tqdm
import torch.utils.tensorboard as tb
import math

import utils
import dataset_imagenet12
import dataset_toybox

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)
TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"


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
    Class definition for the domain adaptation module
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
        self.domain_classifier = nn.Linear(self.fc_size, 1)
    
    def forward(self, x, source_size, alpha):
        """
        Forward method
        """
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        feats_source = feats[:source_size]
        logits = self.classifier(feats_source)
        grad_reversed_out = GradReverse.apply(feats, alpha)
        dom = self.domain_classifier(grad_reversed_out)
        return feats, logits, dom.squeeze()


class Experiment:
    """
    Class used to run the experiments
    """
    
    def __init__(self):
        self.net = Network()
        
        self.optimizer = torch.optim.SGD(self.net.backbone.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9)
        self.optimizer.add_param_group({'params': self.net.classifier.parameters()})
        
        self.optimizer.add_param_group({'params': self.net.domain_classifier.parameters()})
        in12_train_transform = transforms.Compose([  # transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0.5,
                                                   # brightness=0.3),
                                                  transforms.ToPILImage(),
                                                  transforms.Resize(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
                                                  ])
        
        in12_test_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
                                                  ])
        
        tb_transform = transforms.Compose(
            [
                # transforms.ColorJitter(hue=0.2, contrast=0.5, saturation=0.5, brightness=0.3),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)
            ])
        
        self.dataset1 = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True,
                                                             transform=in12_train_transform, fraction=0.1)
        self.dataset2 = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, train=True, transform=tb_transform,
                                                     rng=np.random.default_rng(), num_instances=-1,
                                                     num_images_per_class=500, hypertune=True)
        
        self.loader_1 = torchdata.DataLoader(self.dataset1, batch_size=64, shuffle=True, num_workers=4)
        self.loader_2 = torchdata.DataLoader(self.dataset2, batch_size=64, shuffle=True, num_workers=4)
        
        self.test_dataset1 = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False,
                                                                  transform=in12_test_transform)
        self.test_dataset2 = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, train=False, transform=tb_transform,
                                                          rng=np.random.default_rng(), hypertune=True)
        
        self.test_loader_1 = torchdata.DataLoader(self.test_dataset1, batch_size=64, shuffle=True, num_workers=4)
        self.test_loader_2 = torchdata.DataLoader(self.test_dataset2, batch_size=64, shuffle=True, num_workers=4)
        
        self.net = self.net.cuda()
    
    def train(self):
        """
        Train network
        """
        import datetime
        dt_now = datetime.datetime.now()
        tb_writer = tb.SummaryWriter(log_dir="../runs/dann_tb_temp_" + dt_now.strftime("%b-%d-%Y-%H-%M") + "/")
        loader_2_iter = iter(self.loader_2)
        num_epochs = 10
        self.net.backbone.train()
        self.net.classifier.train()
        self.net.domain_classifier.train()
        total_batches = 0
        for ep in range(1, num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.loader_1)
            for idxs, img1, labels1 in tqdm_bar:
                try:
                    idxs2, img2, labels2 = next(loader_2_iter)
                except StopIteration:
                    loader_2_iter = iter(self.loader_2)
                    idxs2, img2, labels2 = next(loader_2_iter)
                self.optimizer.zero_grad()
                
                images = torch.concat([img1, img2], dim=0)
                dom_labels = torch.cat([torch.zeros(img1.size(0)), torch.ones(img2.size(0))])
                images, dom_labels, labels1 = images.cuda(), dom_labels.cuda(), labels1.cuda()
                
                p = total_batches / (len(self.loader_1) * num_epochs)
                alfa = 2 / (1 + math.exp(-10 * p)) - 1
                
                f, l, d = self.net.forward(images, img1.size(0), alpha=alfa)
                ce_loss = nn.CrossEntropyLoss()(l, labels1)
                total_batches += 1
                dom_pred = torch.sigmoid(d)
                dom_loss = nn.BCELoss()(dom_pred, dom_labels)
                total_loss = ce_loss + dom_loss
                total_loss.backward()
                tqdm_bar.set_description("Ep: {}/{}  LR: {:.4f}  CE Loss: {:.4f}  Dom Loss: {:.4f} Lmbda: {:.4f}  "
                                         "Tot Loss: {:.4f}".format(ep, num_epochs, self.optimizer.param_groups[0]['lr'],
                                                                   ce_loss.item(), dom_loss.item(), alfa,
                                                                   total_loss.item()))
                self.optimizer.step()
                tb_writer.add_scalar(tag="LR", scalar_value=self.optimizer.param_groups[0]['lr'],
                                     global_step=total_batches)
                tb_writer.add_scalar(tag="Lambda", scalar_value=alfa, global_step=total_batches)
                tb_writer.add_scalar(tag="p", scalar_value=p, global_step=total_batches)
                tb_writer.add_scalars(main_tag="Training", tag_scalar_dict={'CE Loss': ce_loss.item(),
                                                                            'Dom Loss': dom_loss.item(),
                                                                            'Total Loss': total_loss.item()
                                                                            },
                                      global_step=total_batches)
            
            if (ep + 1) % 20 == 0:
                self.optimizer.param_groups[0]['lr'] *= 1.0
        tb_writer.close()
    
    def eval(self):
        """
        Eval network on test datasets
        """
        self.net.backbone.eval()
        self.net.classifier.eval()
        self.net.domain_classifier.eval()
        dataloaders = [self.test_loader_1, self.test_loader_2]
        d_names = ['in12', 'toybox']
        for idx in range(2):
            loader = dataloaders[idx]
            total_num = 0
            correct_num = 0
            for idxs, images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    _, logits, _ = self.net.forward(images, images.size(0), alpha=0)
                    res, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                    total_num += logits.size(0)
                    correct_num += res[0].item() * logits.size(0)
            acc = correct_num / total_num
            print("Accuracy on dataset {}: {:.2f}".format(d_names[idx], acc))


if __name__ == "__main__":
    exp = Experiment()
    exp.train()
    exp.eval()


