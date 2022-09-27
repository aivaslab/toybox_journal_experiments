"""
This module exposes classes and methods that can be used to study the results
of different experiments in varying granularity.
"""
import csv
import os

out_dir = "../out/"
imagenet_coco_data_dir = "../data_12/"
toybox_data_dir = "../../toybox_unsupervised_learning/data/"
toybox_classes = ["airplane", "ball", "car", "cat", "cup", "duck", "giraffe", "helicopter", "horse", "mug", "spoon",
                  "truck"]


class BaseEvaluator:
    """
    This class exposes methods common to all derived evaluators
    """
    
    def __init__(self, out_dir_name, dataset_name):
        assert os.path.isdir(out_dir + out_dir_name)
        self.out_dir_path = out_dir + out_dir_name
        self.dataset_name = dataset_name
        self.target_csv_file_path_1 = "".join([self.out_dir_path, "/eval_0_", self.dataset_name, "_test.csv"])
        self.target_csv_file_path_2 = "".join([self.out_dir_path, "/eval_1_", self.dataset_name, "_test.csv"])
        if not os.path.isfile(self.target_csv_file_path_1):
            self.target_csv_file_path_1 = None
        if not os.path.isfile(self.target_csv_file_path_2):
            self.target_csv_file_path_2 = None
    
    @staticmethod
    def eq_op(n1, n2):
        """
        Return True if two vals are equal and vice-versa
        """
        return n1 == n2
    
    @staticmethod
    def neq_op(n1, n2):
        """
        Return True if two values are not equal and vice-versa
        """
        return not n1 == n2
    
    def calc_acc(self):
        """
        This method calculates overall accuracy for the specified CSV
        """
        csv_file_name = self.target_csv_file_path_1
        test_correct = 0
        test_total = 0
        with open(csv_file_name, "r") as csv_file:
            eval_data = list(csv.DictReader(csv_file))
        for i in range(len(eval_data)):
            pred_label = int(eval_data[i]['Predicted Label'])
            true_label = int(eval_data[i]['True Label'])
            if pred_label == true_label:
                test_correct += 1
            test_total += 1
        print("Overall accuracy is {:2.3f}".format(test_correct / test_total * 100))
    
    def get_class_accuracy(self):
        """
        This method calculates the per-class accuracy from the specified CSV
        """
        csv_file_name = self.target_csv_file_path_1
        test_correct = [0 for _ in range(len(toybox_classes))]
        test_total = [0 for _ in range(len(toybox_classes))]
        with open(csv_file_name, "r") as csv_file:
            eval_data = list(csv.DictReader(csv_file))
        for i in range(len(eval_data)):
            pred_label = int(eval_data[i]['Predicted Label'])
            true_label = int(eval_data[i]['True Label'])
            if pred_label == true_label:
                test_correct[true_label] += 1
            test_total[true_label] += 1
        print("==================================================================================")
        print("Printing accuracy for each class")
        print("==================================================================================")
        for i in range(len(toybox_classes)):
            print("{0:12s}:  {1:2.3f}".format(toybox_classes[i], test_correct[i] / test_total[i] * 100))
        print("==================================================================================")
    
    def compare_two_models(self, reverse=False, same=False):
        """
        This method compares which images misclassified in test_csv_1
        were classified correctly in test_csv_2
        """
        if reverse:
            test_csv_1, test_csv_2 = self.target_csv_file_path_2, self.target_csv_file_path_1
        else:
            test_csv_1, test_csv_2 = self.target_csv_file_path_1, self.target_csv_file_path_2
        if same:
            comp_op = self.eq_op
        else:
            comp_op = self.neq_op
        csv_file_1 = open(test_csv_1, "r")
        csv_file_1_data = list(csv.DictReader(csv_file_1))
        csv_file_2 = open(test_csv_2, "r")
        csv_file_2_data = list(csv.DictReader(csv_file_2))
        change_list = []
        for idx_row, row in enumerate(csv_file_1_data):
            idx = int(row['Index'])
            true_label = int(row['True Label'])
            predicted_label = int(row['Predicted Label'])
            if comp_op(true_label, predicted_label):
                idx_found = -1
                for idx_row_2, row_2 in enumerate(csv_file_2_data):
                    idx_2 = int(row_2['Index'])
                    if idx_2 == idx:
                        idx_found = idx_row_2
                        break
                assert idx_found > -1, "Index not found in file 2"
                idx_2 = int(csv_file_2_data[idx_found]['Index'])
                assert idx_2 == idx
                true_label_2 = int(csv_file_2_data[idx_found]['True Label'])
                assert true_label_2 == true_label
                predicted_label_2 = int(csv_file_2_data[idx_found]['Predicted Label'])
                if predicted_label_2 == true_label:
                    change_list.append(idx)
        
        print("{} changes in test files".format(len(change_list)))
        return change_list
    
    def get_acc_by_candidates(self):
        """
        This method calculates accuracy by each candidate category/instance within each of Toybox
        categories. It should be defined by each of the derived classes
        """
        raise NotImplementedError()


class ToyboxEvaluator(BaseEvaluator):
    """
    This class contains all the methods for aggregating results from CSVs containing eval on Toybox dataset.
    """
    
    def __init__(self, out_dir_name):
        super().__init__(out_dir_name=out_dir_name, dataset_name="toybox")
        self.train_data_csv_file_name = toybox_data_dir + "toybox_data_interpolated_cropped_train.csv"
        self.test_data_csv_file_name = toybox_data_dir + "toybox_data_interpolated_cropped_test.csv"
    
    def get_acc_by_candidates(self):
        """
        This method splits the test results on ImageNet+COCO
        by the candidate categories.
        """
        csv_file_name = self.target_csv_file_path_1
        test_correct = {}
        test_total = {}
        test_items = {}
        for cl in toybox_classes:
            test_items[cl] = []
        with open(csv_file_name, "r") as csv_file:
            eval_data = list(csv.DictReader(csv_file))
        with open(self.test_data_csv_file_name, "r") as test_data_file:
            test_data = list(csv.DictReader(test_data_file))
        for i in range(len(eval_data)):
            idx = int(eval_data[i]['Index'])
            cl = test_data[idx]['Class']
            pred_label = int(eval_data[i]['Predicted Label'])
            true_label = int(eval_data[i]['True Label'])
            obj = int(test_data[idx]['Object'])
            if obj not in test_items[cl]:
                test_items[cl].append(obj)
            assert int(true_label) == int(test_data[int(idx)]['Class ID'])
            if (true_label, obj) not in test_correct.keys():
                test_correct[(true_label, obj)] = 0
                test_total[(true_label, obj)] = 0
            if pred_label == true_label:
                test_correct[(true_label, obj)] += 1
            test_total[(true_label, obj)] += 1
        print(test_items)
        print("==================================================================================")
        print("Printing accuracy for each class and object")
        print("==================================================================================")
        print("{0:12s}:  {1:5s}    {2:5s}    {3:5s}    {4:5s}".format("Class", "Obj1", "Obj2", "Obj3", "Total"))
        for i in range(len(toybox_classes)):
            cl = toybox_classes[i]
            corr = [0.0 for i in range(len(test_items['airplane']))]
            tot = 0
            tot_corr = 0
            for j in range(len(test_items['ball'])):
                test_obj = test_items[cl][j]
                tot += test_total[(i, test_obj)]
                tot_corr += test_correct[(i, test_obj)]
                corr[j] = test_correct[(i, test_obj)] / test_total[(i, test_obj)] * 100
            pc_corr = tot_corr / tot * 100
            print("{0:12s}:  {1:2.2f}    {2:2.2f}    {3:2.2f}    {4:2.2f}".format(toybox_classes[i],
                                                                                  corr[0], corr[1], corr[2], pc_corr))
        print("==================================================================================")


class IN12Evaluator(BaseEvaluator):
    """
    This class contains all the methods for aggregating results for CSVs containing eval on IN+COCO-12 dataset.
    """
    
    def __init__(self, out_dir_name):
        super().__init__(out_dir_name=out_dir_name, dataset_name="imagenet_coco")
        self.train_data_csv_file_name = imagenet_coco_data_dir + "train.csv"
        self.test_data_csv_file_name = imagenet_coco_data_dir + "test.csv"
    
    def get_imgnet_coco_split_giraffe(self):
        """
        This method calculates the relative accuracies on the giraffe images
        from the ImageNet and COCO datasets
        """
        csv_file_name = self.target_csv_file_path_1
        assert os.path.isfile(csv_file_name), "File not found:{}".format(csv_file_name)
        csv_file = open(csv_file_name, "r")
        csv_data = list(csv.DictReader(csv_file))
        
        is_train = "train" in csv_file_name
        if is_train:
            data_csv_file_name = imagenet_coco_data_dir + "train.csv"
        else:
            data_csv_file_name = imagenet_coco_data_dir + "test.csv"
        
        assert os.path.isfile(data_csv_file_name), "File not found:{}".format(data_csv_file_name)
        data_csv_file = open(data_csv_file_name, "r")
        detailed_data_csv = list(csv.DictReader(data_csv_file))
        
        in12_total_rows = 0
        in12_correct_rows = 0
        coco_total_rows = 0
        coco_correct_rows = 0
        giraffe_true_label = toybox_classes.index("giraffe")
        for row_idx, row in enumerate(csv_data):
            true_label = int(row['True Label'])
            if true_label == giraffe_true_label:
                predicted_label = int(row['Predicted Label'])
                idx = int(row["Index"])
                file_path = detailed_data_csv[idx]["File Path"]
                is_imagenet = "n02439033" in file_path
                if is_imagenet:
                    in12_total_rows += 1
                    in12_correct_rows += 1 if true_label == predicted_label else 0
                else:
                    coco_total_rows += 1
                    coco_correct_rows += 1 if true_label == predicted_label else 0
        print("COCO accuracy for giraffes:{}".format(coco_correct_rows / coco_total_rows * 100.0))
        print("ImageNet accuracy for giraffes:{}".format(in12_correct_rows / in12_total_rows * 100.0))
        csv_file.close()
        data_csv_file.close()
    
    def get_acc_by_candidates(self):
        """
        This method splits the test results on ImageNet+COCO
        by the candidate categories.
        """
        csv_file_name = self.target_csv_file_path_1
        assert os.path.isfile(csv_file_name), "File not found: {}".format(csv_file_name)
        csv_file = open(csv_file_name, "r")
        csv_data = list(csv.DictReader(csv_file))
        
        is_train = "train" in csv_file_name
        if is_train:
            data_csv_file_name = imagenet_coco_data_dir + "train.csv"
        else:
            data_csv_file_name = imagenet_coco_data_dir + "test.csv"
        assert os.path.isfile(data_csv_file_name), "File not found: {}".format(data_csv_file_name)
        
        data_csv_file = open(data_csv_file_name, "r")
        detailed_data_csv = list(csv.DictReader(data_csv_file))
        uniq_candidate_dict = {}
        total_imgs_cand = {}
        correct_imgs_cand = {}
        for cl_idx, cl in enumerate(toybox_classes):
            uniq_candidate_dict[cl] = []
            total_imgs_cand[cl] = {}
            correct_imgs_cand[cl] = {}
            for row_idx, row in enumerate(csv_data):
                true_label = int(row['True Label'])
                if true_label == cl_idx:
                    img_idx = int(row['Index'])
                    predicted_label = int(row['Predicted Label'])
                    img_data = detailed_data_csv[img_idx]
                    img_path = img_data['File Path']
                    img_path_hier = img_path.split("/")
                    file_name = img_path_hier[-1]
                    tokens = file_name.split("_")
                    if len(tokens) == 1:
                        cand_name = "coco"
                    else:
                        cand_name = tokens[0]
                    if cand_name not in uniq_candidate_dict[cl]:
                        uniq_candidate_dict[cl].append(cand_name)
                        total_imgs_cand[cl][cand_name] = 0
                        correct_imgs_cand[cl][cand_name] = 0
                    total_imgs_cand[cl][cand_name] += 1
                    correct_imgs_cand[cl][cand_name] += 1 if predicted_label == true_label else 0
            print("There are {} candidate classes for class {}: {}".format(str(len(uniq_candidate_dict[cl])), cl,
                                                                           ", ".join(uniq_candidate_dict[cl])))
            acc_list = []
            for cand in uniq_candidate_dict[cl]:
                total_imgs = total_imgs_cand[cl][cand]
                correct_imgs = correct_imgs_cand[cl][cand]
                print("{}: {} {}".format(cand, total_imgs, correct_imgs / total_imgs * 100.0))
                acc_list.append(correct_imgs / total_imgs * 100.0)
            mean = sum(acc_list) / len(acc_list)
            variance = sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)
            print("Class: {} mean: {:.4f}, std: {:.4f}, variance: {:.4f}".format(cl, mean, variance ** 0.5, variance))
