"""
Trying to learn from noisy labels
"""
import argparse
import math
import torch
import torch.utils.data as torchdata
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt

import dataset_imagenet12
import utils

IN12_DATA_PATH = "../data_12/IN-12/"
IN12_MEAN = (0.4570, 0.4873, 0.5008)
IN12_STD = (0.2918, 0.2711, 0.2730)
TB_BASELINE_BB_PATH = "../out/TOYBOX_BASELINE_FINAL/Aug-13-2022-00-00/backbone_trainer_resnet18_backbone.pt"
TB_BASELINE_CL_PATH = "../out/TOYBOX_BASELINE_FINAL/Aug-13-2022-00-00/backbone_trainer_resnet18_classifier.pt"


class Model:
    """
    Model to learn from noisy labels
    """
    
    def __init__(self, exp_args):
        self.args = exp_args
        self.train_fraction = self.args['fraction']
        self.starting_lr = self.args['lr']
        self.num_epochs = self.args['epochs']
        self.toybox_pretrained_model = self.args['toybox']
        
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
        if self.toybox_pretrained_model:
            print("Loading backbone weights from {}".format(TB_BASELINE_BB_PATH))
            self.backbone.load_state_dict(torch.load(TB_BASELINE_BB_PATH))
            print("Loading classifier weights from {}".format(TB_BASELINE_CL_PATH))
            self.classifier.load_state_dict(torch.load(TB_BASELINE_CL_PATH))
        
        tr_trnsfrm = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=256),
                                         transforms.RandomResizedCrop(size=224),
                                         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
        te_trnsfrm = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
        
        train_dataset = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True, transform=tr_trnsfrm,
                                                             fraction=self.train_fraction)
        test_dataset = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False, transform=te_trnsfrm)
        self.train_loader = torchdata.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        self.test_loader = torchdata.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        lr = self.get_lr(batches=0)
        self.optimizer = torch.optim.SGD(self.backbone.parameters(), lr=lr, weight_decay=1e-6)
        self.optimizer.add_param_group({'params': self.classifier.parameters(), 'lr': lr})
        
        self.loaders = [self.train_loader, self.test_loader]
        self.loader_names = ['in12_train', 'in12_test']
    
    def get_lr(self, batches):
        """
        Returns the lr of the current batch
        """
        total_batches = self.num_epochs * len(self.train_loader)
        if batches <= 2 * len(self.train_loader):
            p = (batches + 1) / (2 * len(self.train_loader))
            lr = p * self.starting_lr
        else:
            p = (batches - 2 * len(self.train_loader)) / (total_batches - 2 * len(self.train_loader))
            lr = 0.5 * self.starting_lr * (1 + math.cos(math.pi * p))
        return lr
    
    def train(self):
        """Train the model"""
        self.backbone.cuda()
        self.classifier.cuda()
        self.backbone.train()
        self.classifier.cuda()
        
        total_batches = 0
        for ep in range(1, self.num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.train_loader, ncols=150)
            ep_batches = 0
            ep_ce_loss = 0.0
            for _, images, labels in tqdm_bar:
                self.optimizer.zero_grad()
                
                images, labels = images.cuda(), labels.cuda()
                feats = self.backbone.forward(images)
                feats = feats.view(feats.size(0), -1)
                logits = self.classifier.forward(feats)
                ce_loss = nn.CrossEntropyLoss()(logits, labels)
                
                ce_loss.backward()
                self.optimizer.step()
                ep_batches += 1
                ep_ce_loss += ce_loss.item()
                
                tqdm_bar.set_description("Ep: {}/{}  BLR: {:.3f}  CLR: {:.3f}  CE: {:.4f}".
                                         format(ep, self.num_epochs, self.optimizer.param_groups[0]['lr'],
                                                self.optimizer.param_groups[1]['lr'], ep_ce_loss / ep_batches))
                total_batches += 1
                next_lr = self.get_lr(batches=total_batches)
                for idx in range(len(self.optimizer.param_groups)):
                    self.optimizer.param_groups[idx]['lr'] = next_lr
            tqdm_bar.close()
    
    def eval(self):
        """
        Eval network on test datasets
        """
        self.backbone.eval()
        self.classifier.eval()
        accuracies = {}
        for d_idx in range(len(self.loaders)):
            max_probs = []
            loader = self.loaders[d_idx]
            total_num = 0
            correct_num = 0
            total_bins = [0] * 50
            correct_bins = [0] * 50
            for idx, images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    feats = self.backbone.forward(images)
                    logits = self.classifier.forward(feats)
                    probs = torch.softmax(logits, dim=1)
                    max_prob = torch.max(probs, dim=1)
                    max_probs += list(max_prob.values.cpu().numpy())
                    res, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                    total_num += logits.size(0)
                    correct_num += res[0].item() * logits.size(0)
                    for i in range(images.size(0)):
                        p = max_prob.values[i].item()
                        bin_num = min(int(p / 0.02), 49)
                        total_bins[bin_num] += 1
                        if pred[i] == labels[i]:
                            correct_bins[bin_num] += 1
            acc = correct_num / total_num
            print(len(max_probs), max(max_probs), min(max_probs))
            print("Accuracy on {}: {:.2f}".format(self.loader_names[d_idx], acc))
            accuracies[self.loader_names[d_idx]] = acc
            print(total_bins)
            print(correct_bins, sum(correct_bins) / sum(total_bins))
            print([correct_bins[i] / total_bins[i] if total_bins[i] > 0 else None for i in range(len(total_bins))])


def get_parser():
    """Generate parser for experiment"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", '-e', default=50, type=int)
    parser.add_argument("--fraction", '-f', default=0.1, type=float)
    parser.add_argument("--lr", '-lr', default=0.1, type=float)
    parser.add_argument("--toybox", default=False, action='store_true')
    return vars(parser.parse_args())


if __name__ == "__main__":
    model = Model(exp_args=get_parser())
    model.train()
    model.eval()
