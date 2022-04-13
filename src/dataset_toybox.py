"""
Module exposing class with dataset for preliminary experiments in the Toybox Journal Paper.
"""
import logging

import cv2
import numpy as np
import csv
import pickle
import os
import torchvision.transforms as transforms
import torch.utils.data

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)

toybox_classes = ["airplane", "ball", "car", "cat", "cup", "duck", "giraffe", "helicopter", "horse", "mug", "spoon",
                  "truck"]

toybox_videos = ("rxplus", "rxminus", "ryplus", "ryminus", "rzplus", "rzminus")


class ToyboxDataset(torch.utils.data.Dataset):
    """
    Class for loading Toybox data for classification. The user can specify the number of instances per class
    and the number of images per class. If number of images per class is -1, all images are selected.
    """
    def __init__(self, root, rng, train=True, transform=None, size=224, hypertune=False, num_instances=-1,
                 num_images_per_class=-1, views=toybox_videos):
    
        self.data_path = root
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.size = size
        self.rng = rng
        self.num_instances = num_instances
        self.num_images_per_class = num_images_per_class
        self.views = []
        for view in views:
            assert view in toybox_videos
            self.views.append(view)
        try:
            assert os.path.isdir(self.data_path)
        except AssertionError:
            raise AssertionError("Data directory not found:", self.data_path)
        print(self.views)
        self.label_key = 'Class ID'
        if self.hypertune:
            self.trainImagesFile = self.data_path + "toybox_data_interpolated_cropped_dev.pickle"
            self.trainLabelsFile = self.data_path + "toybox_data_interpolated_cropped_dev.csv"
            self.testImagesFile = self.data_path + "toybox_data_interpolated_cropped_val.pickle"
            self.testLabelsFile = self.data_path + "toybox_data_interpolated_cropped_val.csv"
        else:
            self.trainImagesFile = self.data_path + "toybox_data_interpolated_cropped_train.pickle"
            self.trainLabelsFile = self.data_path + "toybox_data_interpolated_cropped_train.csv"
            self.testImagesFile = self.data_path + "toybox_data_interpolated_cropped_test.pickle"
            self.testLabelsFile = self.data_path + "toybox_data_interpolated_cropped_test.csv"
    
        super().__init__()
    
        if self.train:
            self.indicesSelected = []
            with open(self.trainImagesFile, "rb") as pickleFile:
                self.train_data = pickle.load(pickleFile)
            with open(self.trainLabelsFile, "r") as csvFile:
                self.train_csvFile = list(csv.DictReader(csvFile))
            self.set_train_indices()
            self.verify_train_indices()
            if self.num_images_per_class > 0:
                logging.debug("Loaded dataset with {} instances from each class and {} images from each class "
                              "divided equally amongst chosen instances....".format(self.num_instances,
                                                                                    self.num_images_per_class))
            else:
                logging.debug("Loaded dataset with {} instances from each class and all images from each of the chosen "
                              "instances....".format(self.num_instances))
        else:
            with open(self.testImagesFile, "rb") as pickleFile:
                self.test_data = pickle.load(pickleFile)
            with open(self.testLabelsFile, "r") as csvFile:
                self.test_csvFile = list(csv.DictReader(csvFile))
    
    def __len__(self):
        if self.train:
            return len(self.indicesSelected)
        else:
            return len(self.test_data)
    
    def __getitem__(self, index):
        if self.train:
            actual_index = self.indicesSelected[index]
            img = np.array(cv2.imdecode(self.train_data[actual_index], 3))
            label = int(self.train_csvFile[actual_index][self.label_key])
        else:
            actual_index = index
            img = np.array(cv2.imdecode(self.test_data[index], 3))
            label = int(self.test_csvFile[index][self.label_key])
        
        if self.transform is not None:
            imgs = self.transform(img)
        else:
            imgs = img
        return actual_index, imgs, label
    
    def verify_train_indices(self):
        """
        This method verifies that the indices chosen for training has the same number of instances
        per class as specified in self.num_instances.
        """
        unique_objs = {}
        for idx_selected in self.indicesSelected:
            cl = self.train_csvFile[idx_selected]['Class']
            if cl not in unique_objs.keys():
                unique_objs[cl] = []
            obj = int(self.train_csvFile[idx_selected]['Object'])
            if obj not in unique_objs[cl]:
                unique_objs[cl].append(obj)
            view = self.train_csvFile[idx_selected]['Transformation']
            assert view in self.views
        for cl in toybox_classes:
            assert len(unique_objs[cl]) == self.num_instances
        logging.debug("Verified that all chosen images correspond to {} instances for each "
                      "class....".format(self.num_instances))
        logging.debug("Verified that all images selected come from the specified views...")
            
    def set_train_indices(self):
        """
        This method sets the training indices based on the settings provided in init().
        """
        obj_dict = {}
        obj_id_dict = {}
        for row in self.train_csvFile:
            cl = row['Class']
            if cl not in obj_dict.keys():
                obj_dict[cl] = []
            obj = int(row['Object'])
            if obj not in obj_dict[cl]:
                obj_dict[cl].append(obj)
                obj_start_id = int(row['Obj Start'])
                obj_end_id = int(row['Obj End'])
                obj_id_dict[(cl, obj)] = (obj_start_id, obj_end_id)
        
        if self.num_instances < 0:
            self.num_instances = len(obj_dict['airplane'])
        
        assert self.num_instances <= len(obj_dict['airplane']), "Number of instances must be less than number " \
                                                                "of objects in CSV: {}".format(len(obj_dict['ball']))
        
        if self.num_images_per_class < 0:
            num_images_per_instance = [-1 for _ in range(self.num_instances)]
        else:
            num_images_per_instance = [int(self.num_images_per_class/self.num_instances) for _ in
                                       range(self.num_instances)]
            remaining = max(0, self.num_images_per_class - num_images_per_instance[0] * self.num_instances)
            idx_instance = 0
            while remaining > 0:
                num_images_per_instance[idx_instance] += 1
                idx_instance = (idx_instance + 1) % self.num_instances
                remaining -= 1
        
        for cl in obj_dict.keys():
            obj_list = obj_dict[cl]
            selected_objs = self.rng.choice(obj_list, self.num_instances, replace=False)
            assert len(selected_objs) == len(set(selected_objs))
            for idx_obj, obj in enumerate(selected_objs):
                start_row = obj_id_dict[(cl, obj)][0]
                end_row = obj_id_dict[(cl, obj)][1]
                all_possible_rows = [obj_row for obj_row in range(start_row, end_row + 1)]
                
                rows_with_specified_views = []
                for obj_row in all_possible_rows:
                    view_row = self.train_csvFile[obj_row]['Transformation']
                    if view_row in self.views:
                        rows_with_specified_views.append(obj_row)
                num_images_obj = len(rows_with_specified_views)
                
                num_required_images = num_images_per_instance[idx_obj]
                if num_required_images < 0:
                    num_required_images = num_images_obj
                
                selected_indices_obj = []
                while num_required_images >= num_images_obj:
                    for idx_row in rows_with_specified_views:
                        selected_indices_obj.append(idx_row)
                    num_required_images -= num_images_obj
                additional_rows = self.rng.choice(rows_with_specified_views, num_required_images,
                                                  replace=False)
                assert len(additional_rows) == len(set(additional_rows))
                
                for idx_row in additional_rows:
                    selected_indices_obj.append(idx_row)
                for idx_row in selected_indices_obj:
                    assert start_row <= idx_row <= end_row
                    row_video = self.train_csvFile[idx_row]['Transformation']
                    assert row_video in self.views
                    self.indicesSelected.append(idx_row)


if __name__ == "__main__":
    trnsfrm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=mean, std=std)])
    dataset = ToyboxDataset(root="../../toybox_unsupervised_learning/data/", transform=trnsfrm,
                            rng=np.random.default_rng(0), hypertune=False, train=True, num_instances=10,
                            num_images_per_class=1000, views=("ryminus", "ryplus"))
    print(len(dataset))
