"""
get toybox images for some objects
"""
import pickle
import os
import numpy as np
import cv2

import dataset_toybox

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)
TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)
ALL_DATASETS = ["Toybox", "IN12"]

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"
OUT_DIR = "../IMS_2023/Toybox/Giraffes/"
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    file_name = TOYBOX_DATA_PATH + "toybox_data_interpolated_cropped_dev.pickle"
    assert os.path.isfile(file_name)
    images_file = open(file_name, "rb")
    all_images = pickle.load(images_file)
    start_points = [59000]
    num = 1000
    for st in start_points:
        for idx in range(num):
            img = np.array(cv2.imdecode(all_images[idx+st], 3))
            
            cv2.imwrite(OUT_DIR+"img_"+str(idx+st)+".png", img)
            