"""
Module for model that learns from the target distribution
"""
import torch
import os
import tqdm
import math
import torch.nn as nn
import argparse
import torch.utils.data as torchdata
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt

import utils
import JAN_network
import JAN_dataset

OUT_DIR = "../out/Entropy/"
RUNS_DIR = "../runs/Entropy/"
DATASETS = ['toybox', 'in12']


class Experiment:
    """
    Experiment class for entropy model
    """
    
    def __init__(self, exp_args):
        self.args = exp_args
        self.source_dataset = self.args['dataset_source']
        self.target_dataset = self.args['dataset_target']
        self.num_epochs = self.args['num_epochs']
        self.normalize = not self.args['no_normalize']
        self.lr_anneal = self.args['anneal']
        self.backbone = self.args['backbone']
        self.save = self.args['save']
        self.combined_batch = self.args['combined_batch']
        self.starting_lr = self.args['starting_lr']
        self.hypertune = not self.args['final']
        self.b_size = self.args['batchsize']
        self.debug = self.args['debug']
        
        network_args = {'backbone': self.backbone, 'datasets': [self.source_dataset, self.target_dataset]}
        self.net = JAN_network.get_network(args=network_args)
        dataset_args = {'train': True, 'hypertune': self.hypertune}
        self.dataset1 = JAN_dataset.prepare_dataset(d_name=self.source_dataset, args=dataset_args)
        self.dataset2 = JAN_dataset.prepare_dataset(d_name=self.target_dataset, args=dataset_args)
        print("{} -> {}".format(str(self.dataset1), str(self.dataset2)))
        
        self.loader_1 = torchdata.DataLoader(self.dataset1, batch_size=self.b_size, shuffle=True, num_workers=4,
                                             drop_last=True)
        self.loader_2 = torchdata.DataLoader(self.dataset2, batch_size=self.b_size, shuffle=True, num_workers=4,
                                             drop_last=True)
        if self.debug:
            print(str(self.dataset1), ":", utils.online_mean_and_sd(self.loader_1))
            print(str(self.dataset2), ":", utils.online_mean_and_sd(self.loader_2))
        
        dataset_args['train'] = False
        self.test_dataset1 = JAN_dataset.prepare_dataset(d_name=self.source_dataset, args=dataset_args)
        self.test_dataset2 = JAN_dataset.prepare_dataset(d_name=self.target_dataset, args=dataset_args)
        self.test_loader_1 = torchdata.DataLoader(self.test_dataset1, batch_size=2 * self.b_size, shuffle=False,
                                                  num_workers=4)
        self.test_loader_2 = torchdata.DataLoader(self.test_dataset2, batch_size=2 * self.b_size, shuffle=False,
                                                  num_workers=4)
        
        self.loaders = [self.loader_1, self.test_loader_1, self.loader_2, self.test_loader_2]
        self.loader_names = [self.source_dataset, self.source_dataset + "_test", self.target_dataset,
                             self.target_dataset + "_test"]
        
        self.net = self.net.cuda()
        classifier_lr = self.get_lr(batches=0.0)
        self.optimizer = torch.optim.SGD(self.net.backbone.parameters(), lr=classifier_lr / 10, weight_decay=1e-5,
                                         momentum=0.9)
        self.optimizer.add_param_group({'params': self.net.classifier.parameters(), 'lr': classifier_lr})
        
        import datetime
        self.exp_time = datetime.datetime.now()
        runs_path = RUNS_DIR + self.source_dataset.upper() + "_" + self.target_dataset.upper() + "/exp_" \
                             + self.exp_time.strftime("%b-%d-%Y-%H-%M") + "/"
        self.tb_writer = tb.SummaryWriter(log_dir=runs_path)
        print("Saving experiment tracking data to {}...".format(runs_path))
        self.save_args(path=runs_path)
    
    def save_args(self, path):
        """Save the experiment args in json file"""
        import json
        json_args = json.dumps(self.args)
        out_file = open(path + "exp_args.json", "w")
        out_file.write(json_args)
        out_file.close()
    
    def get_lr(self, batches):
        """
        Returns the lr of the current batch
        """
        total_batches = self.num_epochs * len(self.loader_1)
        if batches <= 2 * len(self.loader_1):
            p = (batches + 1) / (2 * len(self.loader_1))
            lr = p * self.starting_lr
        else:
            p = (batches - 2 * len(self.loader_1)) / (total_batches - 2 * len(self.loader_1))
            lr = 0.5 * self.starting_lr * (1 + math.cos(math.pi * p))
        return lr
    
    @staticmethod
    def entropy_loss(logits):
        """Entropy Loss calculation"""
        logits = torch.softmax(logits, dim=1)
        log_logits = torch.clamp(torch.log2(logits), max=10000, min=-10000)
        entropy = logits * log_logits * -1
        entropy = torch.sum(entropy, dim=1)
        return entropy
    
    def train(self):
        """
        Train network
        """
        self.calc_test_losses(batches=0)
        self.net.backbone.train()
        self.net.classifier.train()
        
        total_batches = 0
        loader_2_iter = iter(self.loader_2)
        for ep in range(1, self.num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.loader_1, ncols=150)
            ep_batches = 0
            ep_ce_loss = 0
            ep_entropy_loss = 0
            ep_tot_loss = 0
            for idx1, img1, labels1 in tqdm_bar:
                try:
                    idx2, img2, labels2 = next(loader_2_iter)
                except StopIteration:
                    loader_2_iter = iter(self.loader_2)
                    idx2, img2, labels2 = next(loader_2_iter)
                self.optimizer.zero_grad()
                alfa = 0.01  # 2 / (1 + math.exp(-10 * p)) - 1
                
                img1, labels1 = img1.cuda(), labels1.cuda()
                img2 = img2.cuda()
                if self.combined_batch:
                    img = torch.concat([img1, img2], dim=0)
                    features, logits = self.net.forward(img)
                    ssize = img1.shape[0]
                    s_f, s_l = features[:ssize], logits[:ssize]
                    t_f, t_l = features[ssize:], logits[ssize:]
                    
                    if total_batches == 0 and self.debug:
                        print(img1.size(), img2.size(), img.size(), s_l.size(), t_l.size(), logits.size(),
                              s_f.size(), t_f.size(), features.size())
                else:
                    s_f, s_l = self.net.forward(img1)
                    t_f, t_l = self.net.forward(img2)
                    if total_batches == 0 and self.debug:
                        print(img1.size(), img2.size(), s_l.size(), t_l.size(), s_f.size(), t_f.size())
                
                ce_loss = nn.CrossEntropyLoss()(s_l, labels1)
                total_batches += 1
                entropy = self.entropy_loss(logits=t_l)
                if total_batches == 0 and self.debug:
                    print(entropy.size())
                entropy = torch.mean(entropy)
                total_loss = ce_loss + alfa * entropy
                total_loss.backward()
                self.optimizer.step()
                
                ep_batches += 1
                ep_ce_loss += ce_loss.item()
                ep_entropy_loss += entropy.item()
                ep_tot_loss += total_loss.item()
                tqdm_bar.set_description("Ep: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE Loss: {:.4f}  Entropy Loss: {:.4f} "
                                         "Lmbda: {:.4f}  "
                                         "Tot Loss: {:.4f}".format(ep, self.num_epochs,
                                                                   self.optimizer.param_groups[0]['lr'],
                                                                   self.optimizer.param_groups[1]['lr'],
                                                                   ep_ce_loss / ep_batches,
                                                                   ep_entropy_loss / ep_batches, alfa,
                                                                   ep_tot_loss / ep_batches))
                
                self.tb_writer.add_scalar(tag="LR", scalar_value=self.optimizer.param_groups[0]['lr'],
                                          global_step=total_batches)
                self.tb_writer.add_scalar(tag="Lambda", scalar_value=alfa, global_step=total_batches)
                self.tb_writer.add_scalars(main_tag="Training",
                                           tag_scalar_dict={'CE Loss (Batch)': ce_loss.item(),
                                                            'Entropy Loss (Batch)': entropy.item(),
                                                            'Total Loss (Batch)': total_loss.item()
                                                            },
                                           global_step=total_batches)
                self.tb_writer.add_scalars(main_tag="Training",
                                           tag_scalar_dict={'CE Loss (Epoch Avg)': ep_ce_loss / ep_batches,
                                                            'Entropy Loss (Epoch Avg)': ep_entropy_loss / ep_batches,
                                                            'Total Loss (Epoch Avg)': ep_tot_loss / ep_batches
                                                            },
                                           global_step=total_batches)
                
                next_lr = self.get_lr(batches=total_batches)
                self.optimizer.param_groups[0]['lr'] = next_lr / 10
                self.optimizer.param_groups[1]['lr'] = next_lr
            
            self.calc_test_losses(batches=total_batches)
            self.net.classifier.train()
            self.net.backbone.train()
        
        if self.save:
            out_dir = OUT_DIR + self.source_dataset.upper() + "_" + self.target_dataset.upper() + "/exp_" + \
                      self.exp_time.strftime("%b-%d-%Y-%H-%M") + "/"
            os.makedirs(out_dir, exist_ok=False)
            print("Saving model components to {}".format(out_dir))
            torch.save(self.net.backbone.state_dict(), out_dir + "backbone_final.pt")
            torch.save(self.net.classifier.state_dict(), out_dir + "classifier_final.pt")
    
    def calc_test_losses(self, batches):
        """
        Pass dataset through network and calculate average entropy
        """
        total_entropy = 0.0
        total_ce = 0.0
        total_batches = 0
        total_examples = 0
        total_activations = None
        self.net.backbone.eval()
        self.net.classifier.eval()
        for _, images, labels in self.test_loader_2:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                f, t_l = self.net.forward(images)
                p = torch.softmax(t_l, dim=1)
                if total_activations is None:
                    total_activations = torch.sum(p, dim=0)
                else:
                    total_activations += torch.sum(p, dim=0)
                total_examples += t_l.shape[0]
                batch_entropy = self.entropy_loss(logits=t_l)
                batch_ce = nn.CrossEntropyLoss()(t_l, labels)
                total_ce += batch_ce.item()
                total_entropy += torch.mean(batch_entropy)
                total_batches += 1
        with torch.no_grad():
            mean_activation = total_activations/total_examples
            mean_activation_entropy = self.entropy_loss(mean_activation.unsqueeze(0))
        
        test_entropy = total_entropy / total_batches
        test_ce = total_ce / total_batches
        self.tb_writer.add_scalar(tag="Test/Entropy", scalar_value=test_entropy, global_step=batches)
        self.tb_writer.add_scalar(tag="Test/CE", scalar_value=test_ce, global_step=batches)
        self.tb_writer.add_scalar(tag="Test/Avg Activation Entropy", scalar_value=mean_activation_entropy,
                                  global_step=batches)
        fig, ax = plt.subplots()
        labels = [i for i in range(12)]
        ax.bar(labels, mean_activation.cpu().numpy(), 0.5)
        ax.set_ylabel('Class Probability')
        ax.set_xlabel('Class Label')
        self.tb_writer.add_figure(tag='Test/Avg Activation', figure=fig, global_step=batches)
    
    def eval(self):
        """
        Eval network on test datasets
        """
        self.net.backbone.eval()
        self.net.classifier.eval()
        accuracies = {}
        for d_idx in range(len(self.loaders)):
            loader = self.loaders[d_idx]
            total_num = 0
            correct_num = 0
            for idx, images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    _, logits = self.net.forward(images)
                    res, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                    total_num += logits.size(0)
                    correct_num += res[0].item() * logits.size(0)
            acc = correct_num / total_num
            print("Accuracy on {}: {:.2f}".format(self.loader_names[d_idx], acc))
            accuracies[self.loader_names[d_idx]] = acc
            self.tb_writer.add_text(tag="Accuracy/" + self.loader_names[d_idx], text_string=str(acc))
        acc_file_path = RUNS_DIR + self.source_dataset.upper() + "_" + self.target_dataset.upper() + "/exp_" \
                                 + self.exp_time.strftime("%b-%d-%Y-%H-%M") + "/acc.json"
        import json
        accuracies_json = json.dumps(accuracies)
        acc_file = open(acc_file_path, "w")
        acc_file.write(accuracies_json)
        acc_file.close()
        
        self.tb_writer.close()


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
    parser.add_argument("--no-normalize", "-nn", default=False, action='store_true')
    parser.add_argument("--backbone", "-bb", default="", type=str)
    parser.add_argument("--save", "-s", default=False, action='store_true')
    parser.add_argument("--combined-batch", "-c", default=False, action='store_true')
    parser.add_argument("--starting-lr", "-lr", default=0.05, type=float)
    parser.add_argument("--final", default=False, action='store_true')
    parser.add_argument("--batchsize", "-b", default=64, type=int)
    parser.add_argument("--debug", default=False, action='store_true')
    
    return vars(parser.parse_args())


if __name__ == "__main__":
    exp = Experiment(exp_args=get_parser())
    exp.train()
    exp.eval()
