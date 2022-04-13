"""
This module extracts images from Coco dataset which
overlap with the Toybox classes.
"""
import os
import json
import shutil

coco_train_dir = "../data_coco/COCO_2017/train2017/"
coco_train_annotations_filename = "../data_coco/COCO_2017/annotations/instances_train2017.json"
coco_val_dir = "../data_coco/COCO_2017/val2017/"
coco_val_annotations_filename = "../data_coco/COCO_2017/annotations/instances_val2017.json"
out_dir = "../data_coco_12/"


def get_cats_dict(cat_dict_from_json):
    """
    This method creates a mapping between the category ids and
    category names in COCO 2017 dataset.
    """
    category_dict = {}
    for category in cat_dict_from_json:
        category_dict[category['id']] = category['name']
    return category_dict


def get_ims_for_cat(anns_dict_from_json, query_cat):
    """
    This method returns a list of the images which correspond to a specific
    category
    """
    ims_list = []
    for ann in anns_dict_from_json:
        ann_id = ann['category_id']
        if ann_id == query_cat:
            ims_list.append(ann['image_id'])
    print(len(ims_list), len(set(ims_list)))
    return set(ims_list)


if __name__ == "__main__":
    train_annotations_file = open(coco_train_annotations_filename)
    train_annotations = json.load(train_annotations_file)
    val_annotations_file = open(coco_val_annotations_filename)
    val_annotations = json.load(val_annotations_file)
    
    cl_name = 'giraffe'
    print(train_annotations.keys())
    cats_dict = get_cats_dict(cat_dict_from_json=train_annotations['categories'])
    query_cl = -1
    for cat_id in cats_dict.keys():
        cat_name = cats_dict[cat_id]
        if cat_name == cl_name:
            query_cl = cat_id
    assert query_cl > 0
    print(cl_name, query_cl)
    
    cat_im_ids = get_ims_for_cat(anns_dict_from_json=train_annotations['annotations'], query_cat=query_cl)
    cl_train_out_dir = out_dir + "train/" + cl_name + "/"
    os.makedirs(cl_train_out_dir, exist_ok=True)
    for im in cat_im_ids:
        im_file_name = coco_train_dir + str(im).zfill(12) + ".jpg"
        try:
            assert os.path.isfile(im_file_name)
        except AssertionError:
            print("File not found:", im_file_name)
        shutil.copy2(im_file_name, cl_train_out_dir)
    
    cat_im_ids = get_ims_for_cat(anns_dict_from_json=val_annotations['annotations'], query_cat=query_cl)
    cl_val_out_dir = out_dir + "val/" + cl_name + "/"
    os.makedirs(cl_val_out_dir, exist_ok=True)
    for im in cat_im_ids:
        im_file_name = coco_val_dir + str(im).zfill(12) + ".jpg"
        try:
            assert os.path.isfile(im_file_name)
        except AssertionError:
            print("File not found:", im_file_name)
        shutil.copy2(im_file_name, cl_val_out_dir)
