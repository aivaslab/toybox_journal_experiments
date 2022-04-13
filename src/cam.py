"""
This module is an implementation of the network viz method,
CAM. https://arxiv.org/pdf/1512.04150.pdf
"""
import csv

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import cv2
import math

import dataloader_toybox_old
import dataset_imagenet12
import parse_config
import Evaluator

out_dir = "../out/"
img_out_dir = "../cam_images_trial/"
imagenet_coco_data_dir = "../data_12/"
toybox_data_dir = "../../toybox_unsupervised_learning/data/"
toybox_mean = (0.3499, 0.4374, 0.5199)
toybox_std = (0.1623, 1894, 1775)


class CAM:
    """
    Class for implementing the CAM method
    """
    
    def __init__(self, dir_name, comp_name):
        self.dir_name = dir_name
        self.comp_name = comp_name
        self.inp = None
        self.outp = None
        self.backbone_file_path = out_dir + dir_name + comp_name + "_resnet18_" + "backbone.pt"
        self.classifier_file_path = out_dir + dir_name + comp_name + "_resnet18_" + "classifier.pt"
        assert os.path.isfile(self.backbone_file_path), "File not found: {}".format(self.backbone_file_path)
        assert os.path.isfile(self.classifier_file_path), "File not found:{}".format(self.classifier_file_path)
        assert "resnet18" in self.backbone_file_path and "resnet18" in self.classifier_file_path
        
        self.backbone = models.resnet18(pretrained=True, num_classes=1000)
        fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(fc_size, 12)
        self.backbone.load_state_dict(torch.load(self.backbone_file_path))
        self.classifier.load_state_dict((torch.load(self.classifier_file_path)))
        self.backbone.layer4[1].register_forward_hook(self.forward_hook)
        self.dataset = None
    
    def forward_hook(self, _, inp, outp):
        """
        Method for forward hook
        """
        self.inp = inp[0]
        self.outp = outp
    
    def build_dataset(self, raw_images, raw_labels):
        """
        Returns images as tensors.
        """
        
        class Dataset:
            """
            Custom dataset class for PyTorch dataloaders
            """
            def __init__(self, images, labels, mean, std):
                self.images = images
                self.labels = labels
                self.trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std)])
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, item):
                img = self.images[item]
                img = self.trnsfrm(img)
                label = self.labels[item]
                return item, img, label
        self.dataset = Dataset(images=raw_images, labels=raw_labels, mean=toybox_mean, std=toybox_std)
    
    def run(self, raw_labels, raw_imgs, img_paths, dataset_name):
        """
        Run CAM
        """
        save_path = "".join([img_out_dir, dataset_name, "/", self.dir_name, "/", self.comp_name, "/"])
        os.makedirs(save_path, exist_ok=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.backbone.cuda()
        self.classifier.cuda()
        self.build_dataset(raw_images=raw_imgs, raw_labels=raw_labels)
        assert self.dataset is not None, "DataLoader is not set"
        
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=False, num_workers=4)
        for idxs, images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            
            self.backbone.forward(images)
            weights_all = self.classifier.weight[labels]
            cam_4_all = torch.sum(weights_all[:, :, None, None] * self.outp, dim=1)
            cam_4_all = torch.clamp(cam_4_all, min=0)
            for id_batch, idx in enumerate(idxs):
                img_path = img_paths[idx]
                img = raw_imgs[idx]
                img = img / 255
                img_shape = img.shape[:2]
                img_cam = cam_4_all[id_batch].cpu().numpy()
                img_cam = img_cam - np.min(img_cam)
                img_cam = img_cam / (1e-8 + np.max(img_cam))
                img_cam = cv2.resize(img_cam, img_shape)
                heatmap = cv2.applyColorMap(np.uint8(255 * img_cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 300
                sum_img = heatmap + img
                sum_img = sum_img / np.max(sum_img)
                sum_img = np.uint8(255 * sum_img)
                img = np.uint8(255 * img)
                two_ims = cv2.hconcat([img, sum_img])
                two_ims = cv2.resize(two_ims, (img.shape[0] * 4, img.shape[0] * 2))
                img_file_name = img_path.split("/")[-1]
                cv2.imwrite(save_path + img_file_name, two_ims)
            
        self.dataset = None


if __name__ == "__main__":
    out_dir_name = "Mar-18-2022-01-09/"
    component_1 = "backbone_trainer"
    component_2 = "classifier_trainer"
    cam_1 = CAM(dir_name=out_dir_name, comp_name=component_1)
    cam_2 = CAM(dir_name=out_dir_name, comp_name=component_2)
    
    in12_evaluator = Evaluator.IN12Evaluator(out_dir_name=out_dir_name)
    change_list_12 = in12_evaluator.compare_two_models(reverse=False, same=False)
    test_file_path = imagenet_coco_data_dir + "test.csv"
    test_file = open(test_file_path, "r")
    test_data_csv = list(csv.DictReader(test_file))
    bsize = 64
    for iter_id in range(0, math.ceil(len(change_list_12) / bsize)):
        all_imgs = []
        all_paths = []
        all_labels = []
        for idx in range(iter_id*bsize, min((iter_id + 1) * bsize, len(change_list_12))):
            img_idx = change_list_12[idx]
            img_filepath = test_data_csv[img_idx]['File Path']
            img_cl_id = int(test_data_csv[img_idx]['Class ID'])
            img = cv2.resize(cv2.imread(img_filepath), (224, 224))
            all_imgs.append(np.array(img))
            all_paths.append(img_filepath)
            all_labels.append(img_cl_id)
        all_imgs = np.stack(all_imgs, axis=0)
        print(all_imgs.shape)
        cam_1.run(raw_labels=all_labels, raw_imgs=all_imgs, img_paths=all_paths, dataset_name="imagenet_test")
        cam_2.run(raw_labels=all_labels, raw_imgs=all_imgs, img_paths=all_paths, dataset_name="imagenet_test")
        