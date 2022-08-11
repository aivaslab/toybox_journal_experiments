"""Code for mean teacher adaptation from tb to in12"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
import torch.utils.data as torchdata
import torch.nn.functional as functional
import torchvision.models as models
import csv
import argparse
import numpy as np

import utils
import dataset_toybox
import dataset_ssl_in12
import dataset_imagenet12

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)
TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)
ALL_DATASETS = ["Toybox", "IN12"]

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"


class Network(nn.Module):
    """ResNet-18 backbone for toybox and in12 training"""
    
    def __init__(self, pretrained):
        super(Network, self).__init__()
        self.pretrained = pretrained
        self.backbone = models.resnet18(pretrained=self.pretrained)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
        
    def forward(self, x):
        """forward for module"""
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
    
    def set_train_mode(self):
        """set network in train mode"""
        self.backbone.train()
        self.classifier.train()
        
    def set_eval_mode(self):
        """set network in eval mode"""
        self.backbone.eval()
        self.classifier.eval()
        
    
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
        self.student = Network(pretrained=False)
        self.teacher = Network(pretrained=False)
        self.teacher.backbone.load_state_dict(self.student.backbone.state_dict())
        self.teacher.classifier.load_state_dict(self.student.classifier.state_dict())
        self.tb_train_transform = self.get_train_transform(mean=TOYBOX_MEAN, std=TOYBOX_STD)
        self.in12_train_transform = self.get_train_transform(mean=IN12_MEAN, std=IN12_STD)
        self.in12_test_transform = self.get_test_transform(mean=IN12_MEAN, std=IN12_STD, ten_crop=False)
        
        self.source_dataset = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, train=True, hypertune=True,
                                                           transform=self.tb_train_transform, num_instances=-1,
                                                           num_images_per_class=2000, rng=np.random.default_rng(0)
                                                           )
        self.target_dataset = dataset_ssl_in12.DatasetIN12(fraction=0.5, hypertune=True,
                                                           transform=self.in12_train_transform)
        self.target_dataset_sup = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, fraction=0.5,
                                                                       transform=self.in12_test_transform)
        
        self.source_loader = torchdata.DataLoader(self.source_dataset, batch_size=64, shuffle=True, num_workers=2,
                                                  pin_memory=True, persistent_workers=False)
        self.target_loader = torchdata.DataLoader(self.target_dataset, batch_size=64, shuffle=True, num_workers=2,
                                                  pin_memory=True, persistent_workers=False)

        self.target_loader_sup = torchdata.DataLoader(self.target_dataset_sup, batch_size=64, shuffle=True,
                                                      num_workers=2, pin_memory=True, persistent_workers=False)

    @staticmethod
    def get_train_transform(mean, std):
        """
        Returns the train_transform parameterized by the mean and std of current dataset.
        """
        prob = 0.2
        color_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=prob),
                            transforms.RandomApply([transforms.ColorJitter(hue=0.2)], p=prob),
                            transforms.RandomApply([transforms.ColorJitter(saturation=0.2)], p=prob),
                            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=prob),
                            transforms.RandomEqualize(p=prob),
                            transforms.RandomPosterize(bits=4, p=prob),
                            transforms.RandomAutocontrast(p=prob)
                            ]
    
        trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)),
                                      transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0),
                                                                   interpolation=transforms.InterpolationMode.BICUBIC),
                                      transforms.RandomOrder(color_transforms),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std),
                                      transforms.RandomErasing(p=0.5)])
    
        return trnsfrm

    @staticmethod
    def get_test_transform(mean, std, ten_crop=True):
        """
        Returns the test transform parameterized by the mean and std of current dataset.
        """
        if ten_crop:
            trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)),
                                          transforms.TenCrop(size=224),
                                          transforms.Lambda(lambda crops:
                                                            torch.stack([transforms.ToTensor()(crop) for crop in crops])
                                                            ),
                                          transforms.Normalize(mean, std)])
        else:
            trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                          transforms.ToTensor(), transforms.Normalize(mean, std)])
    
        return trnsfrm
    
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
        source_loader_iter = iter(self.source_loader)
        for epoch in range(1, num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.target_loader, ncols=150)
            for indices2, (images2_1, images2_2) in tqdm_bar:
                try:
                    indices, images, labels = next(source_loader_iter)
                except StopIteration:
                    source_loader_iter = iter(self.source_loader)
                    indices, images, labels = next(source_loader_iter)
                # print(images.shape, images2_1.shape, images2_2.shape)
                optimizer.zero_grad()
                images, labels = images.cuda(), labels.cuda()
                images2_1, images2_2 = images2_1.cuda(), images2_2.cuda()
                
                preds = self.student.forward(images)
                cls_loss = nn.CrossEntropyLoss()(preds, labels)
                
                target_logits_student = torch.softmax(self.student.forward(images2_1), dim=1)
                with torch.no_grad():
                    target_logits_teacher = torch.softmax(self.teacher.forward(images2_2), dim=1)
                
                self_loss, _, _ = compute_aug_loss(target_logits_student, target_logits_teacher)
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
            loader = self.target_loader_sup
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
