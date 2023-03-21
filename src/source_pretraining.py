"""Pretrain model with the source dataset"""
import argparse
import os
import tqdm

import torch
import torchvision.models as models
import torch.utils.data as torchdata
import torch.nn as nn


import datasets
import utils

OUT_PATH = "../out/source_pretraining_models/"
os.makedirs(OUT_PATH, exist_ok=True)


class Trainer:
    """Class definition for model that pretrains on source dataset"""
    def __init__(self, args):
        self.args = args
        
        self.network = models.resnet18(weights=None)
        self.net_fc_size = self.network.fc.in_features
        self.network.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(self.net_fc_size, 31))
        dataset_args = {'train': True,
                        'hypertune': True,
                        'special_aug': False,
                        'pair': False,
                        'fraction': 1.0
                        }
        self.train_data = datasets.prepare_dataset(d_name=self.args['dataset'], args=dataset_args)
        self.train_loader = torchdata.DataLoader(self.train_data, batch_size=64, shuffle=True, num_workers=4)
        print("Training on {:s} with size {}".format(str(self.train_data), len(self.train_data)))
        
        dataset_args['train'] = False
        self.test_data = datasets.prepare_dataset(d_name=self.args['dataset'], args=dataset_args)
        self.test_loader = torchdata.DataLoader(self.test_data, batch_size=128, shuffle=False, num_workers=4)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args['lr'])
        self.optimizer.add_param_group({'params': self.classifier.parameters(), 'lr': self.args['lr']})
        
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                                                                  total_iters=2*len(self.train_loader))
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                           T_max=(self.args['epochs'] - 2) *
                                                                           len(self.train_loader))
        self.combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[2*len(self.train_loader) + 1]
        )
        self.network.cuda()
        self.classifier.cuda()
        
    def set_model_train(self):
        """Set model in training mode"""
        for params in self.network.parameters():
            params.requires_grad = True
        for params in self.classifier.parameters():
            params.requires_grad = True
        self.network.train()
        self.classifier.train()

    def set_model_eval(self):
        """Set model in eval mode"""
        for params in self.network.parameters():
            params.requires_grad = False
        for params in self.classifier.parameters():
            params.requires_grad = False
        self.network.eval()
        self.classifier.eval()

    def train(self):
        """Training method"""
        total_batches = 0
        for ep in range(1, self.args['epochs'] + 1):
            tqdmBar = tqdm.tqdm(total=len(self.train_loader))
            ep_batches = 0
            ep_total_loss = 0.0
            self.set_model_train()
            for b_id, (idxs, images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                images, labels = images.cuda(), labels.cuda()
                feats = self.network.forward(images)
                logits = self.classifier.forward(feats)
                
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.combined_scheduler.step()
                
                ep_batches += 1
                ep_total_loss += loss.item()
                ep_avg_loss = ep_total_loss / ep_batches
                total_batches += 1
                tqdmBar.set_description("Ep: {}/{}  BLR: {:.4f}  CLR: {:.4f}  Loss: {:.4f}".
                                        format(ep, self.args['epochs'], self.optimizer.param_groups[0]['lr'],
                                               self.optimizer.param_groups[1]['lr'],
                                               ep_avg_loss
                                               )
                                        )
                tqdmBar.update(1)
            tqdmBar.close()
            if ep % 20 == 0:
                self.eval()
        save_dict = {
            'backbone': self.network.state_dict(),
            'classifier': self.classifier.state_dict(),
        }
        torch.save(save_dict, OUT_PATH + "temp.pt")
    
    def eval(self):
        """Eval method"""
        self.set_model_eval()
        
        accs = []
        loaders = [self.train_loader, self.test_loader]
        for loader in loaders:
            n_total = 0
            n_correct = 0
            for _, (idxs, images, labels) in enumerate(loader):
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    feats = self.network.forward(images)
                    logits = self.classifier.forward(feats)
                res, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                n_total += idxs.size(0)
                n_correct += int(res[0].item() * idxs.size(0))
            acc = 1. * (n_correct / n_total)
            accs.append(acc)
           
        print("Train Acc: {:.2f}   Test Acc: {:.2f}".format(accs[0], accs[1]))
        
        
def get_parser():
    """Return parser for pretraining experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", "-d", required=True, choices=['amazon', 'dslr', 'webcam'])
    parser.add_argument("--epochs", "-e", default=50, type=int)
    parser.add_argument("--lr", "-lr", default=0.05, type=float)
    return parser.parse_args()
    
    
def main():
    """Main method"""
    args = vars(get_parser())
    trainer = Trainer(args=args)
    trainer.train()
    trainer.eval()
    
    
if __name__ == "__main__":
    main()
