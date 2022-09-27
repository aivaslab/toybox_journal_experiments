"""get IN classes in IN12"""
import csv

if __name__ == "__main__":
    filename = "../data_12/IN-12/train.csv"
    f_ptr = open(filename, "r")
    csv_file = csv.DictReader(f_ptr)
    all_rows = list(csv_file)
    print(len(all_rows))
    cl_dict = {}
    for idx, row in enumerate(all_rows):
        cl = row['Class']
        src_img_name = row['File Path']
        # print(src_img_name)
        splits = src_img_name.split('/')
        f_name = splits[-1]
        im_cl = f_name.split("_")
        # print(im_cl)
        if cl not in cl_dict:
            cl_dict[cl] = set()
        if len(im_cl) == 2:
            cl_dict[cl].add(im_cl[0])
        else:
            cl_dict[cl].add('coco')
    for key in cl_dict.keys():
        cls = [k for k in cl_dict[key]]
        print(key, ", ".join(cls))