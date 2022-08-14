"""
Module for implementing the Mean Teacher model from the VisDA-2017 winning submission.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
import torch.utils.data as torchdata
import torch.nn.functional as functional
import csv
import argparse
import datetime
import os
import torch.utils.tensorboard as tb

import dataset_mnist_svhn
import utils
import visda_aug
import network_mnist_svhn
import dataset_mnist50
import dataset_svhn_balanced

MNIST_MEAN = (0.1309, 0.1309, 0.1309)
MNIST_STD = (0.2893, 0.2893, 0.2893)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

OUT_DIR = "../out/MNIST50_SVHN_ENSEMBLE/"
RUNS_DIR = "../runs/MNIST50_SVHN_ENSEMBLE/"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


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


class MeanTeacher:
    """
    Module for implementing the mean teacher architecture
    """
    def __init__(self):
        self.student = network_mnist_svhn.Network(n_classes=10)
        self.teacher = network_mnist_svhn.Network(n_classes=10)
        self.teacher.load_state_dict(self.student.state_dict())
        self.mnist_train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(3),
                                                         # transforms.RandomInvert(p=0.5),
                                                         # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                                                         # transforms.ColorJitter(brightness=0.4, contrast=0.2),
                                                         transforms.Pad(2),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
        self.svhn_train_transform = transforms.Compose([transforms.Resize(32),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD)])
        
        self.source_dataset = dataset_mnist50.DatasetMNIST50(root="../data/", train=True,
                                                             transform=self.mnist_train_transform)
        self.target_dataset = dataset_svhn_balanced.BalancedSVHN(root="../data/", train=True, hypertune=True,
                                                                 transform=self.svhn_train_transform)
        self.source_loader = torchdata.DataLoader(self.source_dataset, batch_size=256, shuffle=True, num_workers=4,
                                                  pin_memory=True, persistent_workers=False)
        self.target_loader = torchdata.DataLoader(self.target_dataset, batch_size=256, shuffle=True, num_workers=4,
                                                  pin_memory=True, persistent_workers=False)
    
    def update_teacher(self, alpha=0.99):
        """EMA update for the teacher's weights"""
        for current_params, moving_params in zip(self.teacher.parameters(), self.student.parameters()):
            current_weight, moving_weight = current_params.data, moving_params.data
            current_params.data = current_weight * alpha + moving_weight * (1 - alpha)
    
    def train(self, train_args):
        """Train the network"""
        for params in self.student.parameters():
            params.requires_grad = True
        for params in self.teacher.parameters():
            params.requires_grad = False
        
        self.student.set_train_mode()
        self.teacher.set_train_mode()
        self.student.cuda()
        self.teacher.cuda()
        
        student_params_total = sum(p.numel() for p in self.student.parameters())
        student_params_train = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        print(
            "{}/{} parameters in the student network are trainable".format(student_params_train, student_params_total))
        
        teacher_params_total = sum(p.numel() for p in self.teacher.parameters())
        teacher_params_train = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        print(
            "{}/{} parameters in the teacher network are trainable".format(teacher_params_train, teacher_params_total))
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=train_args['lr'], weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=train_args['epochs'] * len(self.target_loader))
        num_epochs = train_args['epochs']
        aug = visda_aug.get_aug_for_mnist()
        source_loader_iter = iter(self.source_loader)
        dt_now = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M")

        for epoch in range(1, num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.target_loader, ncols=150)
            for indices2, images2, labels2 in tqdm_bar:
                try:
                    indices, images, labels = next(source_loader_iter)
                except StopIteration:
                    source_loader_iter = iter(self.source_loader)
                    indices, images, labels = next(source_loader_iter)
                
                optimizer.zero_grad()
                images = aug.augment(images)
                images, labels = images.cuda(), labels.cuda()
                images2_1, images2_2 = aug.augment_pair(images2)
                images2_1, images2_2 = images2_1.cuda(), images2_2.cuda()
                
                preds = self.student.forward(images)
                cls_loss = nn.CrossEntropyLoss()(preds, labels)
                
                target_logits_student = torch.softmax(self.student.forward(images2_1), dim=1)
                with torch.no_grad():
                    target_logits_teacher = torch.softmax(self.teacher.forward(images2_2), dim=1)
                
                self_loss, _, _ = compute_aug_loss(target_logits_student, target_logits_teacher)
                # self_loss = nn.MSELoss()(target_logits_student, target_logits_teacher)
                unsup_weight = 3.0
                loss = cls_loss + self_loss * unsup_weight
                loss.backward()
                optimizer.step()
                self.update_teacher()
                scheduler.step()
                tqdm_bar.set_description("Epoch: {}/{}  CLS Loss: {:.4f}  SELF Loss: {:.4f}  Total: {:.4f}  LR: {:.4f}".
                                         format(epoch, num_epochs, cls_loss.item(), self_loss.item(),
                                                loss.item(), optimizer.param_groups[0]['lr']))
            tqdm_bar.close()
            if epoch % 10 == 0:
                acc = self.eval_model(training=True)
                print("Source accuracy:{:.2f}".format(acc))
                acc = self.eval_model(training=False)
                print("Target accuracy:{:.2f}".format(acc))
                os.makedirs(OUT_DIR + dt_now, exist_ok=True)
                teacher_file_name = OUT_DIR + dt_now + "/teacher_epoch_" + str(epoch) + ".pt"
                print("Saving teacher network to {}".format(teacher_file_name))
                torch.save(self.teacher.state_dict(), teacher_file_name)

                student_file_name = OUT_DIR + dt_now + "/student_epoch_" + str(epoch) + ".pt"
                print("Saving student network to {}".format(student_file_name))
                torch.save(self.student.state_dict(), student_file_name)

        teacher_file_name = OUT_DIR + dt_now + "/teacher_final.pt"
        print("Saving teacher network to {}".format(teacher_file_name))
        torch.save(self.teacher.state_dict(), teacher_file_name)

        student_file_name = OUT_DIR + dt_now + "/student_final.pt"
        print("Saving student network to {}".format(student_file_name))
        torch.save(self.student.state_dict(), student_file_name)

    def eval_model(self, training=False, csv_file_name=None):
        """
        This method calculates the accuracies on the provided
        dataloader for the current model defined by backbone
        and classifier.
        """
        self.teacher.eval()
        
        top1acc = 0
        tot_train_points = 0
    
        save_csv_file = None
        csv_writer = None
        if csv_file_name is not None:
            save_csv_file = open(csv_file_name, "w")
            csv_writer = csv.writer(save_csv_file)
            csv_writer.writerow(["Index", "True Label", "Predicted Label"])
    
        if training:
            loader = self.source_loader
        else:
            loader = self.target_loader
        # Iterate over batches and calculate top-1 accuracy
        for _, (indices, images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            if len(images.size()) == 5:
                b_size, n_crops, c, h, w = images.size()
            else:
                b_size, c, h, w = images.size()
                n_crops = 1
            with torch.no_grad():
                logits = self.teacher.forward(images)
                logits_avg = logits.view(b_size, n_crops, -1).mean(dim=1)
            top, pred = utils.calc_accuracy(logits_avg, labels, topk=(1,))
            top1acc += top[0].item() * pred.shape[0]
            tot_train_points += pred.shape[0]
            if csv_file_name is not None:
                indices, labels, pred = indices.cpu().numpy(), labels.cpu().numpy(), pred.cpu().numpy()
                for idx in range(pred.shape[0]):
                    csv_writer.writerow([indices[idx], labels[idx], pred[idx]])
        top1acc /= tot_train_points
        if csv_file_name is not None:
            save_csv_file.close()
        self.teacher.train()
        return top1acc
            

class Experiment:
    """
    Class to set up and run MeanTeacher experiments
    """
    
    def __init__(self):
        self.trainer = MeanTeacher()
    
    def run(self, exp_args):
        """Run the experiment with the specified parameters"""
        self.trainer.train(train_args=exp_args)
        source_acc = self.trainer.eval_model(training=True)
        target_acc = self.trainer.eval_model(training=False)
        print("Source Accuracy: {:.2f}".format(source_acc))
        print("Target Accuracy: {:.2f}".format(target_acc))


def get_parser():
    """Parser with arguments for experiment"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", default=100, type=int)
    parser.add_argument("--lr", "-lr", default=0.1, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    exp = Experiment()
    args = vars(get_parser())
    exp.run(exp_args=args)
