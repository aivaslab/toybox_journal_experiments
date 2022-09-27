"""ss"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch
import os

import visda_aug

OUT_DIR = "../IMS_2023/SVHN/"
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    dataset = datasets.SVHN(root="../data/", split='train', transform=None)
    for idx in range(100):
        img, label = dataset[idx]
        f_name = OUT_DIR + str(idx) + ".png"
        img.save(f_name)
        
        aug = visda_aug.get_aug_for_mnist()
        img = transforms.ToTensor()(img)
        img = torch.unsqueeze(img, dim=0)
        img = aug.augment(img)
        img = transforms.ToPILImage()(img[0])

        out_f_name = OUT_DIR + str(idx) + "_aug.png"
        # print(img.shape)
        img.save(out_f_name)
    # for idx in range(50):
    #     f_name = OUT_DIR + str(idx) + ".png"
    #     im = transforms.ToPILImage()(images[idx])
    #     im.save(f_name)
    #     trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(3), transforms.ToTensor()])
    #     img = trnsfrm(images[idx])
    #     img = torch.unsqueeze(img, dim=0)
    #     # print(img.shape)
    #     img = aug.augment(img)
    #     # print(img.shape)
    #     out_f_name = OUT_DIR + str(idx) + "_aug.png"
    #     img = transforms.ToPILImage()(img[0])
    #     img.save(out_f_name)

