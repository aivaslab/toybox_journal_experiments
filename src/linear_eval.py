"""
Module for linear evaluation of pretrained networks
"""

import torch
import torchvision.transforms as transforms
import argparse
import torch.utils.data as torchdata
import torch.utils.tensorboard as tb
import torch.nn as nn
import tqdm
import math
import csv

import LinearEvalNet
import dataset_imagenet12
import utils


IN12_MEAN = (0.4541, 0.4845, 0.4980)
IN12_STD = (0.2928, 0.2738, 0.2756)
IN12_DATA_PATH = "../data_12/IN-12/"

RUNS_DIR = "../runs/LinearEval/"


class Experiment:
    """Class with linear evaluation methods"""
    NETWORKS = {'object': LinearEvalNet.ResNet18WithActivations}
    
    def __init__(self, args):
        self.experiment = args['experiment']
        self.num_epochs = args['epochs']
        self.starting_lr = args['lr']
        self.layer = args['layer']
        self.args = args
        self.backbone = args['backbone']
        self.args['backbone'] = self.backbone
        self.debug = args['debug']
        
        if self.experiment == 'digit':
            raise NotImplementedError("Linear eval for digit classification experiments not implemented yet")
        self.backbone = self.NETWORKS[self.experiment](backbone_file=self.backbone)
        for params in self.backbone.parameters():
            params.requires_grad = False
        assert self.layer in self.backbone.MODULES
        self.classifier = nn.Linear(in_features=self.backbone.FC_SIZES[self.layer], out_features=12)
        
        train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256),
                                              transforms.RandomResizedCrop(size=224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
        
        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
        self.train_dataset = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True,
                                                                  transform=train_transform, fraction=1.0)
        self.test_dataset = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False,
                                                                 transform=test_transform)
        self.train_loader = torchdata.DataLoader(self.train_dataset, batch_size=256, shuffle=True, num_workers=4)
        self.test_loader = torchdata.DataLoader(self.test_dataset, batch_size=256, shuffle=True, num_workers=4)
        self.loaders = [self.train_loader, self.test_loader]
        self.loader_names = ['in12_train', 'in12_test']
        
        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.starting_lr)

        import datetime
        self.exp_time = datetime.datetime.now()
        self.runs_path = RUNS_DIR + self.experiment.upper() + "/exp_" + self.exp_time.strftime("%b-%d-%Y-%H-%M") + "/"
        self.tb_writer = tb.SummaryWriter(log_dir=self.runs_path)
        self.save_args()
        
        self.backbone.cuda()
        self.classifier.cuda()
        self.backbone.eval()
        self.classifier.train()
        
    def save_args(self):
        """Save the experiment args in json file"""
        import json
        json_args = json.dumps(self.args)
        out_file = open(self.runs_path + "exp_args.json", "w")
        out_file.write(json_args)
        out_file.close()
    
    def get_lr(self, p):
        """Retrieve lr for next batch"""
        return 0.5 * self.starting_lr * (1 + math.cos(math.pi * p))
    
    def train(self):
        """Train the method"""
        total_batches = 0
        for ep in range(1, self.num_epochs + 1):
            tqdm_bar = tqdm.tqdm(total=len(self.train_loader))
            ep_batches = 0
            ep_loss = 0.0
            for idx, img, labels in self.train_loader:
                idx, img, labels = idx.cuda(), img.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    self.backbone.forward(img)
                    reduced_feats = self.backbone.get_activation(self.layer)
                if total_batches == 0 and self.debug:
                    print(reduced_feats.shape)
                logits = self.classifier.forward(reduced_feats)
                batch_loss = nn.CrossEntropyLoss()(logits, labels)
                
                batch_loss.backward()
                self.optimizer.step()
                ep_batches += 1
                ep_loss += batch_loss.item()
                tqdm_bar.set_description("Ep: {}/{}  LR: {:.4f}  CE: {:.4f}".format(
                    ep, self.num_epochs, self.optimizer.param_groups[0]['lr'], ep_loss / ep_batches
                ))
                self.tb_writer.add_scalars(main_tag='Training',
                                           tag_scalar_dict={
                                               'LR': self.optimizer.param_groups[0]['lr'],
                                               'Batch CE': batch_loss.item(),
                                               'Epoch CE Avg': ep_loss / ep_batches
                                           },
                                           global_step=total_batches)
                tqdm_bar.update(n=1)
                total_batches += 1
                p = total_batches / (len(self.train_loader) * self.num_epochs)
                next_lr = self.get_lr(p=p)
                self.optimizer.param_groups[0]['lr'] = next_lr
            tqdm_bar.close()

    def eval(self):
        """
        Eval network on test datasets
        """
        self.backbone.eval()
        self.classifier.eval()
        accuracies = {}
        for d_idx in range(len(self.loaders)):
            loader = self.loaders[d_idx]
            total_num = 0
            correct_num = 0
            csv_file_name = self.runs_path + self.loader_names[d_idx] + "_predictions.csv"
            save_csv_file = open(csv_file_name, "w")
            csv_writer = csv.writer(save_csv_file)
            csv_writer.writerow(["Index", "True Label", "Predicted Label"])
        
            for idx, images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    self.backbone.forward(images)
                    activations = self.backbone.get_activation(self.layer)
                    logits = self.classifier.forward(activations)
                    res, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                    total_num += logits.size(0)
                    correct_num += res[0].item() * logits.size(0)
                idx, labels, pred = idx.cpu().numpy(), labels.cpu().numpy(), pred.cpu().numpy()
                for ind in range(pred.shape[0]):
                    csv_writer.writerow([idx[ind], labels[ind], pred[ind]])
        
            acc = correct_num / total_num
            print("Accuracy on {}: {:.2f}".format(self.loader_names[d_idx], acc))
            accuracies[self.loader_names[d_idx]] = acc
            save_csv_file.close()
    
        acc_file_path = self.runs_path + "/acc.json"
        import json
        accuracies_json = json.dumps(accuracies)
        acc_file = open(acc_file_path, "w")
        acc_file.write(accuracies_json)
        acc_file.close()
    
        self.tb_writer.close()
            
            
def get_parser():
    """Parser for linear eval experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", '-exp', required=True, choices=['object'])
    parser.add_argument("--epochs", '-e', default=50, type=int)
    parser.add_argument("--lr", '-lr', default=0.1, type=float)
    parser.add_argument("--layer", '-l', required=True, type=str)
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--backbone", "-bb", required=True, type=str)
    return vars(parser.parse_args())


def main():
    """Main method"""
    exp_args = get_parser()
    exp = Experiment(args=exp_args)
    exp.train()
    exp.eval()
    

if __name__ == "__main__":
    main()
    