"""sdnfndas"""
import os
import cv2
import torchvision.transforms as transforms
import numpy as np

TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)


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
                                  # transforms.Normalize(mean, std),
                                  transforms.RandomErasing(p=0.5),
                                  transforms.ToPILImage()
                                  ])
    
    return trnsfrm


if __name__ == "__main__":
    dir_name = "../IMS_2023/IN12-selected/"
    print(os.listdir(dir_name))
    for file in os.listdir(dir_name):
        if file.endswith(".JPEG") and not file.startswith("aug"):
            f_name = dir_name + file
            img = cv2.imread(f_name)
            # cv2.imshow("sd", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            transform = get_train_transform(mean=TOYBOX_MEAN, std=TOYBOX_STD)
            img = np.array(img)
            # img = np.expand_dims(img, axis=0)
            for idx in range(3):
                tr_img = transform(np.array(img))
                out_file_name = "aug_" + str(idx) + "_" + file
                tr_img.save(dir_name + out_file_name)
            
