"""
Module for implementing the method described in "Self-ensembling for visual domain adaptation"
Self Ensembling Network: https://arxiv.org/pdf/1706.05208.pdf
"""

import torch
import torch.utils.data as torchdata
import torch.nn as nn
import tqdm
import torch.utils.tensorboard as tb
import math
import argparse
import os

import utils
import self_ensemble_network
import DANN_dataset

OUT_DIR = "../out/Self-Ensemble/"
RUNS_DIR = "../runs/Self-Ensemble/"
DATASETS = ['mnist50', 'svhn-b']


def robust_binary_crossentropy(pred, tgt):
    """sdf"""
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def compute_aug_loss(stu_out, tea_out):
    """ Augmentation loss"""
    confidence_thresh = 0.968
    conf_tea = torch.max(tea_out, 1)[0]
    unsup_mask = conf_mask = (conf_tea > confidence_thresh).float()
    unsup_mask_count = conf_mask_count = conf_mask.sum()
    
    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss
    
    # Class balance scaling
    n_samples = unsup_mask.sum()
    avg_pred = n_samples / float(10)
    bal_scale = avg_pred / torch.clamp(tea_out.sum(dim=0), min=1.0)
    bal_scale = bal_scale.detach()
    aug_loss = aug_loss * bal_scale[None, :]
    
    aug_loss = aug_loss.mean(dim=1)
    n_classes = 10
    cls_bal_fn = robust_binary_crossentropy
    unsup_loss = (aug_loss * unsup_mask).mean()
    cls_balance = 0.05
    rampup = 0
    # Class balance loss
    if cls_balance > 0.0:
        # Compute per-sample average predicated probability
        # Average over samples to get average class prediction
        avg_cls_prob = stu_out.mean(dim=0)
        # Compute loss
        equalise_cls_loss = cls_bal_fn(avg_cls_prob, float(1.0 / n_classes))
        
        equalise_cls_loss = equalise_cls_loss.mean() * n_classes
        
        if rampup == 0:
            equalise_cls_loss = equalise_cls_loss * unsup_mask.mean(dim=0)
        
        unsup_loss += equalise_cls_loss * cls_balance
    
    return unsup_loss, conf_mask_count, unsup_mask_count


class Experiment:
    """
    Class used to run the experiments
    """
    
    def __init__(self, exp_args):
        self.args = exp_args
        self.source_dataset = self.args['dataset_source']
        self.target_dataset = self.args['dataset_target']
        self.num_epochs = self.args['num_epochs']
        self.backbone = self.args['backbone']
        self.save = self.args['save']
        self.combined_batch = self.args['combined_batch']
        self.starting_lr = self.args['starting_lr']
        self.hypertune = not self.args['final']
        self.b_size = self.args['batchsize']
        self.debug = self.args['debug']
        
        self.teacher = self_ensemble_network.Network(n_classes=10)
        self.student = self_ensemble_network.Network(n_classes=10)
        self.teacher.load_state_dict(self.student.state_dict())
        for params in self.teacher.parameters():
            params.requires_grad = False
        for params in self.student.parameters():
            params.requires_grad = True
        
        dataset_args = {'train': True, 'hypertune': self.hypertune, 'normalize': True}
        self.dataset1 = DANN_dataset.prepare_dataset(d_name=self.source_dataset, args=dataset_args)
        self.dataset2 = DANN_dataset.prepare_dataset(d_name=self.target_dataset, args=dataset_args)
        print("{} -> {}".format(str(self.dataset1), str(self.dataset2)))
        
        self.loader_1 = torchdata.DataLoader(self.dataset1, batch_size=self.b_size, shuffle=True, num_workers=4,
                                             drop_last=True)
        self.loader_2 = torchdata.DataLoader(self.dataset2, batch_size=self.b_size, shuffle=True, num_workers=4,
                                             drop_last=True)
        dataset_args['train'] = False
        self.test_dataset1 = DANN_dataset.prepare_dataset(d_name=self.source_dataset, args=dataset_args)
        self.test_dataset2 = DANN_dataset.prepare_dataset(d_name=self.target_dataset, args=dataset_args)
        self.test_loader_1 = torchdata.DataLoader(self.test_dataset1, batch_size=2 * self.b_size, shuffle=False,
                                                  num_workers=4)
        self.test_loader_2 = torchdata.DataLoader(self.test_dataset2, batch_size=2 * self.b_size, shuffle=False,
                                                  num_workers=4)
        
        self.loaders = [self.loader_1, self.test_loader_1, self.loader_2, self.test_loader_2]
        self.loader_names = [self.source_dataset, self.source_dataset + "_test", self.target_dataset,
                             self.target_dataset + "_test"]
        
        self.teacher = self.teacher.cuda()
        self.student = self.student.cuda()
        init_lr = self.get_lr(batches=0)
        self.optimizer = torch.optim.SGD(self.student.parameters(), lr=init_lr, weight_decay=1e-5, momentum=0.9)
        
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
        total_batches = self.num_epochs * len(self.loader_2)
        if batches <= 2 * len(self.loader_2):
            p = (batches + 1) / (2 * len(self.loader_2))
            lr = p * self.starting_lr
        else:
            p = (batches - 2 * len(self.loader_2)) / (total_batches - 2 * len(self.loader_2))
            lr = 0.5 * self.starting_lr * (1 + math.cos(math.pi * p))
        return lr

    def update_teacher(self, alpha=0.99):
        """EMA update for the teacher's weights"""
        for current_params, moving_params in zip(self.teacher.parameters(), self.student.parameters()):
            current_weight, moving_weight = current_params.data, moving_params.data
            current_params.data = current_weight * alpha + moving_weight * (1 - alpha)

    def train(self):
        """
        Train network
        """
        self.calc_test_losses(batches=0)
        self.teacher.set_train_mode()
        self.student.set_train_mode()
        
        total_batches = 0
        loader_1_iter = iter(self.loader_1)
        for ep in range(1, self.num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.loader_2, ncols=150)
            ep_batches = 0
            ep_ce_loss = 0
            ep_self_loss = 0
            ep_tot_loss = 0
            for idx2, img2, labels2 in tqdm_bar:
                try:
                    idx1, img1, labels1 = next(loader_1_iter)
                except StopIteration:
                    loader_1_iter = iter(self.loader_1)
                    idx1, img1, labels1 = next(loader_1_iter)
                self.optimizer.zero_grad()
                p = total_batches / (len(self.loader_2) * self.num_epochs)
                alfa = 3  # 2 / (1 + math.exp(-10 * p)) - 1
                
                img1, labels1 = img1.cuda(), labels1.cuda()
                img2, labels2 = img2.cuda(), labels2.cuda()
                if self.combined_batch:
                    img = torch.concat([img1, img2], dim=0)
                    logits = self.student.forward(img)
                    ssize = img1.shape[0]
                    s_l = logits[:ssize]
                    t_l = logits[ssize:]
                    
                    if total_batches == 0 and self.debug:
                        print(img1.size(), img2.size(), img.size(), s_l.size(), logits.size())
                else:
                    s_l = self.student.forward(img1)
                    t_l = self.student.forward(img2)
                    if total_batches == 0 and self.debug:
                        print(img1.size(), img2.size(), s_l.size(), t_l.size())
                
                target_probs = torch.softmax(t_l, dim=1)
                with torch.no_grad():
                    teacher_logits = self.teacher.forward(img2)
                    teacher_probs = torch.softmax(teacher_logits, dim=1)
                
                ce_loss = nn.CrossEntropyLoss()(s_l, labels1)
                total_batches += 1
                self_loss, _, _ = compute_aug_loss(target_probs, teacher_probs)
                total_loss = ce_loss + alfa * self_loss
                total_loss.backward()
                self.optimizer.step()
                self.update_teacher()
                
                with torch.no_grad():
                    val_ce_loss = nn.CrossEntropyLoss()(teacher_logits, labels2)
                    
                ep_batches += 1
                ep_ce_loss += ce_loss.item()
                ep_self_loss += self_loss.item()
                ep_tot_loss += total_loss.item()
                tqdm_bar.set_description("Ep: {}/{}  LR: {:.4f}  CE: {:.3f}  Self-L: {:.3f} "
                                         "Lmbda: {:.2f}  Trgt CE: {:.2f}  "
                                         "Tot Loss: {:.3f}".format(ep, self.num_epochs,
                                                                   self.optimizer.param_groups[0]['lr'],
                                                                   ep_ce_loss / ep_batches, ep_self_loss / ep_batches,
                                                                   alfa,
                                                                   val_ce_loss.item(), ep_tot_loss / ep_batches))
                
                self.tb_writer.add_scalar(tag="LR", scalar_value=self.optimizer.param_groups[0]['lr'],
                                          global_step=total_batches)
                self.tb_writer.add_scalar(tag="Lambda", scalar_value=alfa, global_step=total_batches)
                self.tb_writer.add_scalar(tag="p", scalar_value=p, global_step=total_batches)
                self.tb_writer.add_scalars(main_tag="Training",
                                           tag_scalar_dict={'CE Loss (Batch)': ce_loss.item(),
                                                            'Self Loss (Batch)': self_loss.item(),
                                                            'Total Loss (Batch)': total_loss.item(),
                                                            'Target CE (Batch)': val_ce_loss.item()
                                                            },
                                           global_step=total_batches)
                self.tb_writer.add_scalars(main_tag="Training",
                                           tag_scalar_dict={'CE Loss (Epoch Avg)': ep_ce_loss / ep_batches,
                                                            'Self Loss (Epoch Avg)': ep_self_loss / ep_batches,
                                                            'Total Loss (Epoch Avg)': ep_tot_loss / ep_batches
                                                            },
                                           global_step=total_batches)
                
                next_lr = self.get_lr(batches=total_batches)
                self.optimizer.param_groups[0]['lr'] = next_lr
            
            if ep % 5 == 0:
                self.calc_test_losses(batches=total_batches)
                self.teacher.set_train_mode()
        
        if self.save:
            out_dir = OUT_DIR + self.source_dataset.upper() + "_" + self.target_dataset.upper() + "/exp_" + \
                      self.exp_time.strftime("%b-%d-%Y-%H-%M") + "/"
            os.makedirs(out_dir, exist_ok=False)
            print("Saving model components to {}".format(out_dir))
            torch.save(self.teacher.state_dict(), out_dir + "teacher_final.pt")
            torch.save(self.student.state_dict(), out_dir + "student_final.pt")
    
    def calc_test_losses(self, batches):
        """
        Pass dataset through network and calculate losses
        """
        total_ce = 0.0
        batches_total = 0
        self.teacher.set_eval_mode()
        
        for _, images, labels in self.test_loader_2:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                t_l = self.teacher.forward(images)
                batch_ce = nn.CrossEntropyLoss()(t_l, labels)
                total_ce += batch_ce.item()
                batches_total += 1
        test_ce = total_ce / batches_total
        
        self.tb_writer.add_scalar(tag="Test/CE", scalar_value=test_ce, global_step=batches)
    
    def eval(self):
        """
        Eval network on test datasets
        """
        self.teacher.set_eval_mode()
        accuracies = {}
        for d_idx in range(len(self.loaders)):
            loader = self.loaders[d_idx]
            total_num = 0
            correct_num = 0
            for idx, images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    logits = self.teacher.forward(images)
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
