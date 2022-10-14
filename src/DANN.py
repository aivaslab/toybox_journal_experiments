"""
Module for implementing the DANN algorithm
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
import math
import argparse

import utils
import DANN_network as DANNNetworks
import DANN_dataset as DANNDatasets

DATASETS = ['mnist', 'mnist-m', 'svhn', 'in12', 'toybox']


class Experiment:
    """
    Class used to run the experiments
    """
    
    def __init__(self, exp_args):
        self.args = exp_args
        self.source_dataset = self.args['dataset_source']
        self.target_dataset = self.args['dataset_target']
        self.num_epochs = self.args['num_epochs']
        self.normalize = self.args['normalize']
        self.lr_anneal = self.args['anneal']
        
        self.net = DANNNetworks.get_network(source_dataset=self.source_dataset, target_dataset=self.target_dataset)
        
        dataset_args = {'normalize': self.normalize,
                        'train': True,
                        }
        self.dataset1 = DANNDatasets.prepare_dataset(self.source_dataset, args=dataset_args)
        self.dataset2 = DANNDatasets.prepare_dataset(self.target_dataset, args=dataset_args)
        
        self.loader_1 = torchdata.DataLoader(self.dataset1, batch_size=64, shuffle=True, num_workers=4)
        self.loader_2 = torchdata.DataLoader(self.dataset2, batch_size=64, shuffle=True, num_workers=4)
        
        dataset_args['train'] = False
        self.test_dataset1 = DANNDatasets.prepare_dataset(self.source_dataset, args=dataset_args)
        self.test_dataset2 = DANNDatasets.prepare_dataset(self.target_dataset, args=dataset_args)
        
        self.test_loader_1 = torchdata.DataLoader(self.test_dataset1, batch_size=128, shuffle=True, num_workers=4)
        self.test_loader_2 = torchdata.DataLoader(self.test_dataset2, batch_size=128, shuffle=True, num_workers=4)
        
        self.net = self.net.cuda()
        self.optimizer = torch.optim.SGD(self.net.backbone.parameters(), lr=self.get_lr(p=0.0), weight_decay=1e-5,
                                         momentum=0.9)
        self.optimizer.add_param_group({'params': self.net.classifier.parameters()})
        self.optimizer.add_param_group({'params': self.net.domain_classifier.parameters()})
    
    @staticmethod
    def get_lr(p):
        """
        Returns the lr of the current batch
        """
        mu_0 = 0.01
        alpha = 10
        beta = 0.75
        lr = mu_0 / ((1 + alpha * p) ** beta)
        return lr
    
    def train(self):
        """
        Train network
        """
        import datetime
        dt_now = datetime.datetime.now()
        # tb_writer = tb.SummaryWriter(log_dir="../runs/dann/SVHN_MNIST_TRIAL/exp_" + dt_now.strftime("%b-%d-%Y-%H-%M") +
        #                                      "/")
        loader_2_iter = iter(self.loader_2)
        self.net.backbone.train()
        self.net.classifier.train()
        self.net.domain_classifier.train()
        total_batches = 0
        for ep in range(1, self.num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.loader_1, ncols=150)
            for idx1, img1, labels1 in tqdm_bar:
                try:
                    idx2, img2, labels2 = next(loader_2_iter)
                except StopIteration:
                    loader_2_iter = iter(self.loader_2)
                    idx2, img2, labels2 = next(loader_2_iter)
                self.optimizer.zero_grad()
                
                images = torch.concat([img1, img2], dim=0)
                dom_labels = torch.cat([torch.zeros(img1.size(0)), torch.ones(img2.size(0))])
                images, dom_labels, labels1 = images.cuda(), dom_labels.cuda(), labels1.cuda()
                
                p = total_batches / (len(self.loader_1) * self.num_epochs)
                alfa = 2 / (1 + math.exp(-10 * p)) - 1
                f, l, d = self.net.forward(images, img1.size(0), alpha=alfa)
                
                ce_loss = nn.CrossEntropyLoss()(l, labels1)
                total_batches += 1
                dom_pred = torch.sigmoid(d)
                dom_loss = nn.BCELoss()(dom_pred, dom_labels)
                total_loss = ce_loss + dom_loss
                total_loss.backward()
                tqdm_bar.set_description("Ep: {}/{}  LR: {:.4f}  CE Loss: {:.4f}  Dom Loss: {:.4f} Lmbda: {:.4f}  "
                                         "Tot Loss: {:.4f}".format(ep, self.num_epochs,
                                                                   self.optimizer.param_groups[0]['lr'],
                                                                   ce_loss.item(), dom_loss.item(), alfa,
                                                                   total_loss.item()))
                self.optimizer.step()
                # tb_writer.add_scalar(tag="LR", scalar_value=self.optimizer.param_groups[0]['lr'],
                #                      global_step=total_batches)
                # tb_writer.add_scalar(tag="Lambda", scalar_value=alfa, global_step=total_batches)
                # tb_writer.add_scalar(tag="p", scalar_value=p, global_step=total_batches)
                # tb_writer.add_scalars(main_tag="Training", tag_scalar_dict={'CE Loss': ce_loss.item(),
                #                                                             'Dom Loss': dom_loss.item(),
                #                                                             'Total Loss': total_loss.item()
                #                                                             },
                #                       global_step=total_batches)
                
                if self.lr_anneal:
                    self.optimizer.param_groups[0]['lr'] = self.get_lr(p=p)
        # tb_writer.close()
    
    def eval(self):
        """
        Eval network on test datasets
        """
        self.net.backbone.eval()
        self.net.classifier.eval()
        self.net.domain_classifier.eval()
        dataloaders = [self.loader_1, self.test_loader_1, self.loader_2, self.test_loader_2]
        d_names = [self.source_dataset + "_train", self.source_dataset + "_test", self.target_dataset + "_train",
                   self.target_dataset + "_test"]
        for d_idx in range(len(dataloaders)):
            loader = dataloaders[d_idx]
            total_num = 0
            correct_num = 0
            for idx, images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    _, logits, _ = self.net.forward(images, images.size(0), alpha=0)
                    res, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                    total_num += logits.size(0)
                    correct_num += res[0].item() * logits.size(0)
            acc = correct_num / total_num
            print("Accuracy on dataset {}: {:.2f}".format(d_names[d_idx], acc))


def get_parser():
    """
    Return parser for experiment
    :return:
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_source", "-d1", required=True, choices=DATASETS)
    parser.add_argument("--dataset_target", "-d2", required=True, choices=DATASETS)
    parser.add_argument("--num-epochs", "-e", default=100, type=int)
    parser.add_argument("--anneal", "-a", default=False, action='store_true')
    parser.add_argument("--normalize", "-n", default=False, action='store_true')
    
    return vars(parser.parse_args())


if __name__ == "__main__":
    exp = Experiment(exp_args=get_parser())
    exp.train()
    exp.eval()
