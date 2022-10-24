"""
Module for implementing the JAN algorithm
Joint Adaptation Networks: https://arxiv.org/pdf/1605.06636.pdf
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import torch.nn as nn
import tqdm
import torch.utils.tensorboard as tb
import math
import argparse
import os
import tllib.alignment.jan as jan
import tllib.modules.kernels as kernels
import torch.nn.functional as func

import utils
import JAN_dataset

OUT_DIR = "../out/JAN/"
RUNS_DIR = "../runs/JAN/"
DATASETS = ['amazon', 'dslr', 'webcam', 'toybox', 'in12']


class JANNet(torch.nn.Module):
    """Network for JAN experiments"""
    def __init__(self):
        super(JANNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 31)
        
    def forward(self, x):
        """
        Forward method for network
        """
        feats = self.backbone(x)
        feats_view = feats.view(feats.size(0), -1)
        logits = self.classifier(feats_view)
        return feats, logits


class Experiment:
    """
    Class used to run the experiments
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
        
        self.net = JANNet()
        dataset_args = {'train': True, 'hypertune': self.hypertune}
        self.dataset1 = JAN_dataset.prepare_dataset(d_name=self.source_dataset, args=dataset_args)
        self.dataset2 = JAN_dataset.prepare_dataset(d_name=self.target_dataset, args=dataset_args)
        print("{} -> {}".format(str(self.dataset1), str(self.dataset2)))

        self.loader_1 = torchdata.DataLoader(self.dataset1, batch_size=self.b_size, shuffle=True, num_workers=4,
                                             drop_last=True)
        self.loader_2 = torchdata.DataLoader(self.dataset2, batch_size=self.b_size, shuffle=True, num_workers=4,
                                             drop_last=True)
        dataset_args['train'] = False
        self.test_dataset1 = JAN_dataset.prepare_dataset(d_name=self.source_dataset, args=dataset_args)
        self.test_dataset2 = JAN_dataset.prepare_dataset(d_name=self.target_dataset, args=dataset_args)
        self.test_loader_1 = torchdata.DataLoader(self.test_dataset1, batch_size=2*self.b_size, shuffle=False,
                                                  num_workers=4)
        self.test_loader_2 = torchdata.DataLoader(self.test_dataset2, batch_size=2*self.b_size, shuffle=False,
                                                  num_workers=4)
        
        self.loaders = [self.loader_1, self.test_loader_1, self.loader_2, self.test_loader_2]
        self.loader_names = [self.source_dataset, self.source_dataset + "_test", self.target_dataset,
                             self.target_dataset + "_test"]

        self.net = self.net.cuda()
        classifier_lr = self.get_lr(p=0.0)
        self.optimizer = torch.optim.SGD(self.net.backbone.parameters(), lr=classifier_lr/10, weight_decay=1e-5,
                                         momentum=0.9)
        self.optimizer.add_param_group({'params': self.net.classifier.parameters(), 'lr': classifier_lr})
        self.jmmd_loss = jan.JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([kernels.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                     (kernels. GaussianKernel(sigma=0.92, track_running_stats=False),)),
            linear=False,
            thetas=None
        ).cuda()

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

    def get_lr(self, p):
        """
        Returns the lr of the current batch
        """
        if 'toybox' not in self.source_dataset and 'toybox' not in self.target_dataset:
            mu_0, alpha, beta = 0.01, 10, 0.75
            if self.lr_anneal:
                lr = mu_0 / ((1 + alpha * p) ** beta)
            else:
                lr = mu_0
        else:
            lr = 0.5 * self.starting_lr * (1 + math.cos(math.pi * p))
        return lr

    def train(self):
        """
        Train network
        """
        self.net.backbone.train()
        self.net.classifier.train()

        total_batches = 0
        loader_2_iter = iter(self.loader_2)
        for ep in range(1, self.num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.loader_1, ncols=150)
            ep_batches = 0
            ep_ce_loss = 0
            ep_jmmd_loss = 0
            ep_tot_loss = 0
            for idx1, img1, labels1 in tqdm_bar:
                try:
                    idx2, img2, labels2 = next(loader_2_iter)
                except StopIteration:
                    loader_2_iter = iter(self.loader_2)
                    idx2, img2, labels2 = next(loader_2_iter)
                self.optimizer.zero_grad()
                p = total_batches / (len(self.loader_1) * self.num_epochs)
                alfa = 2 / (1 + math.exp(-10 * p)) - 1

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
                jmmd_loss = self.jmmd_loss((s_f, func.softmax(s_l, dim=1)), (t_f, func.softmax(t_l, dim=1)))
                total_loss = ce_loss + alfa * jmmd_loss
                total_loss.backward()
                self.optimizer.step()

                ep_batches += 1
                ep_ce_loss += ce_loss.item()
                ep_jmmd_loss += jmmd_loss.item()
                ep_tot_loss += total_loss.item()
                tqdm_bar.set_description("Ep: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE Loss: {:.4f}  JMMD Loss: {:.4f} "
                                         "Lmbda: {:.4f}  "
                                         "Tot Loss: {:.4f}".format(ep, self.num_epochs,
                                                                   self.optimizer.param_groups[0]['lr'],
                                                                   self.optimizer.param_groups[1]['lr'],
                                                                   ep_ce_loss/ep_batches, ep_jmmd_loss/ep_batches, alfa,
                                                                   ep_tot_loss/ep_batches))

                self.tb_writer.add_scalar(tag="LR", scalar_value=self.optimizer.param_groups[0]['lr'],
                                          global_step=total_batches)
                self.tb_writer.add_scalar(tag="Lambda", scalar_value=alfa, global_step=total_batches)
                self.tb_writer.add_scalar(tag="p", scalar_value=p, global_step=total_batches)
                self.tb_writer.add_scalars(main_tag="Training",
                                           tag_scalar_dict={'CE Loss (Batch)': ce_loss.item(),
                                                            'JMMD Loss (Batch)': jmmd_loss.item(),
                                                            'Total Loss (Batch)': total_loss.item()
                                                            },
                                           global_step=total_batches)
                self.tb_writer.add_scalars(main_tag="Training",
                                           tag_scalar_dict={'CE Loss (Epoch Avg)': ep_ce_loss / ep_batches,
                                                            'JMMD Loss (Epoch Avg)': ep_jmmd_loss / ep_batches,
                                                            'Total Loss (Epoch Avg)': ep_tot_loss / ep_batches
                                                            },
                                           global_step=total_batches)

                next_lr = self.get_lr(p=p)
                self.optimizer.param_groups[0]['lr'] = next_lr / 10
                self.optimizer.param_groups[1]['lr'] = next_lr

        if self.save:
            out_dir = OUT_DIR + self.source_dataset.upper() + "_" + self.target_dataset.upper() + "/exp_" + \
                      self.exp_time.strftime("%b-%d-%Y-%H-%M") + "/"
            os.makedirs(out_dir, exist_ok=False)
            print("Saving model components to {}".format(out_dir))
            torch.save(self.net.backbone.state_dict(), out_dir + "backbone_final.pt")
            torch.save(self.net.classifier.state_dict(), out_dir + "classifier_final.pt")

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
