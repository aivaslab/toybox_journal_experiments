"""
This module provides the entry code for the supervised experiments
for the Toybox Journal paper.
Usage: ./model.py --log-level=info --config-file=default_config.yaml
Change config YAML file(default_config.yaml by default) to change
experiments.
Config files should be located in ../configs.
"""
import csv
import os
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import torch
from tqdm import tqdm as tqdm
import yaml
import logging
import datetime
import torch.utils.tensorboard as tb
import torch.utils.data
import pytorch_model_summary

import dataset_toybox
import dataset_core50
import dataset_imagenet12
import parse_config
import utils

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)
TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)
CORE50_MEAN = (0.5980, 0.5622, 0.5363)
CORE50_STD = (0.2102, 0.2199, 0.2338)
ALL_DATASETS = ["Toybox", "IN12", "CORE50"]

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"
CORE50_DATA_PATH = "../../toybox-representation-learning/data/"

OUTPUT_DIR = "../out/"
CONFIG_DIR = "../configs/"

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}
LOG_FORMAT_TERMINAL = '%(asctime)s:' + COLOR['GREEN'] + '%(filename)s' + COLOR['ENDC'] + ':%(lineno)s:' + COLOR['RED'] \
                      + '%(levelname)s' + COLOR['ENDC'] + ': %(message)s'
LOG_FORMAT_FILE = '%(asctime)s:%(filename)s:%(lineno)s:%(levelname)s:%(message)s'


class Experiment:
    """
    Class for running experiments. To run experiments, initialize an
    object with required configuration. Different experiments to be run
    in sequence are called components. Call run_components() after
    initialization to run all the components.
    """
    
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.backbone = None
        self.fc_size = -1
        self.classifier = None
        self.hypertune = config_dict['hypertune']
        self.prepare_model()
        self.components = self.config_dict['components']
        self.global_params = self.config_dict['global_params']
        
        self.loader = None
        self.img_aug = None
        self.run_args = None
        self.test_loader = None
        self.test_loader_name = None
        self.num_test_iter = 0
        self.run_id = 0
        
        self.prepare_run()
    
    def prepare_model(self):
        """
        This method prepares the network for experiments. Loads the backbone
        architecture and replaces the number of output neurons. If pretrained
        network has been specified or filename provided, this method loads
        weights from that file. Note that two files are required: one each
        for the backbone and classifier.
        """
        # Initialize backbone and classifier for model and then load
        # weights from corresponding files, if provided.
        logger.info("Loading model backbone: %s and fc layer....", self.config_dict['model']['name'])
        self.backbone = parse_config.get_model_name(models, self.config_dict['model']['name'],
                                                    **self.config_dict['model']['args'])
        logger.debug(pytorch_model_summary.summary(self.backbone, torch.zeros((1, 3, 224, 224)), show_input=True))
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
        if 'load_dir' in self.config_dict['model'].keys():
            
            # TODO: make it possible for user to specify files directly instead of load_dir and load_comp
            
            if self.config_dict['model']['load_dir'] != "":
                
                # Check if load directory exist
                load_dir = OUTPUT_DIR + self.config_dict['model']['load_dir'] + "/"
                logger.info("Trying to load models from %s", load_dir)
                utils.tryAssert(os.path.isdir, load_dir, "Directory not found {}".format(load_dir))
                
                load_comp = self.config_dict['model']['load_comp']
                if load_comp == "":
                    logger.warning("Empty load_comp in config_file. Proceeding with training from scratch....")
                else:
                    # Check if load files exist. If both exist, load weights into initialized networks.
                    backbone_file_name = load_dir + load_comp + "_" + self.config_dict['model']['name'] + "_backbone.pt"
                    logger.info("Loading backbone weights from %s", backbone_file_name)
                    utils.tryAssert(os.path.isfile, backbone_file_name, "File not found {}".format(backbone_file_name))
                    classifier_file_name = load_dir + load_comp + "_" + self.config_dict['model']['name'] \
                                           + "_classifier.pt"
                    logger.info("Loading classifier weights from %s", classifier_file_name)
                    utils.tryAssert(os.path.isfile, classifier_file_name, "File not found {}".format(
                        classifier_file_name))
                    self.backbone.load_state_dict(torch.load(backbone_file_name))
                    self.classifier.load_state_dict(torch.load(classifier_file_name))
            else:
                logger.warning("Empty load_dir in config_file. Proceeding with training from scratch....")
        
        # Move network weights to GPU
        gpu = 0
        torch.cuda.set_device(gpu)
        self.backbone.cuda(gpu)
        self.classifier.cuda(gpu)
    
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
    
    def prepare_model_for_run(self):
        """
        This method prepares the backbone and classifier for a training run.
        """
        tr_args = self.run_args
        assert 'layers_frozen' in tr_args.keys()
        layers_frozen = tr_args['layers_frozen']
    
        # Set which weights will be updated according to config.
        # Set train() for network components. Only matters for regularization
        # techniques like dropout and batchnorm.
        # Set optimizer and add which weights are to be optimized acc. to config.
        for params in self.classifier.parameters():
            params.requires_grad = True
        for params in self.backbone.parameters():
            params.requires_grad = True
    
        self.classifier.train()
        optimizer = parse_config.get_optimizer_name(torch.optim, tr_args['optimizer'], self.classifier.parameters(),
                                                    **tr_args['optimizer_args'])
        if layers_frozen > -1:
            for params in self.backbone.conv1.parameters():
                params.requires_grad = False
            for params in self.backbone.bn1.parameters():
                params.requires_grad = False
            for params in self.backbone.relu.parameters():
                params.requires_grad = False
            for params in self.backbone.maxpool.parameters():
                params.requires_grad = False
            logger.info("Freezing conv1, bn1, relu, maxpool...")
        else:
            optimizer.add_param_group({'params': self.backbone.conv1.parameters()})
            optimizer.add_param_group({'params': self.backbone.bn1.parameters()})
            optimizer.add_param_group({'params': self.backbone.relu.parameters()})
            optimizer.add_param_group({'params': self.backbone.maxpool.parameters()})
            logger.info("Adding conv1, bn1, relu, maxpool to optimizer...")
        layer_mode = False if layers_frozen > -1 else True
        self.backbone.conv1.train(mode=layer_mode)
        self.backbone.bn1.train(mode=layer_mode)
        self.backbone.relu.train(mode=layer_mode)
        self.backbone.maxpool.train(mode=layer_mode)
    
        if layers_frozen > 0:
            for params in self.backbone.layer1.parameters():
                params.requires_grad = False
            logger.info("Freezing layer1")
        else:
            optimizer.add_param_group({'params': self.backbone.layer1.parameters()})
            logger.info("Adding layer1 to optimizer")
        layer_mode = False if layers_frozen > 0 else True
        self.backbone.layer1.train(mode=layer_mode)
    
        if layers_frozen > 1:
            for params in self.backbone.layer2.parameters():
                params.requires_grad = False
            logger.info("Freezing layer2")
        else:
            optimizer.add_param_group({'params': self.backbone.layer2.parameters()})
            logger.info("Adding layer2 to optimizer")
        layer_mode = False if layers_frozen > 1 else True
        self.backbone.layer2.train(mode=layer_mode)
    
        if layers_frozen > 2:
            for params in self.backbone.layer3.parameters():
                params.requires_grad = False
            logger.info("Freezing layer3")
        else:
            optimizer.add_param_group({'params': self.backbone.layer3.parameters()})
            logger.info("Adding layer3 to optimizer")
        layer_mode = False if layers_frozen > 2 else True
        self.backbone.layer3.train(mode=layer_mode)
    
        if layers_frozen > 3:
            for params in self.backbone.layer4.parameters():
                params.requires_grad = False
            for params in self.backbone.avgpool.parameters():
                params.requires_grad = False
            logger.info("Freezing layer4, avgpool")
        else:
            optimizer.add_param_group({'params': self.backbone.layer4.parameters()})
            optimizer.add_param_group({'params': self.backbone.avgpool.parameters()})
            logger.info("Adding layer4, avgpool to optimizer")
        layer_mode = False if layers_frozen > 3 else True
        self.backbone.layer4.train(mode=layer_mode)
        self.backbone.avgpool.train(mode=layer_mode)
    
        return optimizer
        
    def train_model(self):
        """
        Method to train the model for a train component in experiment.
        Assumes that the network has already been initialized to
        starting weights using prepare_model() or by the last component
        in experiment.
        """
        comp_name = self.components[self.run_id]
        tr_args = self.run_args
        if tr_args['num_epochs'] == 0:
            # No training required if number of epochs is 0
            return
        
        optimizer = self.prepare_model_for_run()
        total_params = sum(p.numel() for p in self.backbone.parameters())
        train_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info("{}/{} parameters in the backbone are trainable...".format(train_params, total_params))
        total_params = sum(p.numel() for p in self.classifier.parameters())
        train_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        logger.info("{}/{} parameters in the classifier are trainable...".format(train_params, total_params))
        # Set lr scheduler for training experiment.
        if 'lr_scheduler' in tr_args.keys():
            scheduler = parse_config.get_scheduler_name(torch.optim.lr_scheduler, tr_args['lr_scheduler'], optimizer,
                                                        **tr_args['lr_scheduler_args'])
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
    
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                                             total_iters=2*len(self.loader)-1)
    
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                                   schedulers=[warmup_scheduler, scheduler],
                                                                   milestones=[2*len(self.loader)-1])
    
        num_epochs = tr_args['num_epochs']
        avg_loss = 0.0
        total_batches = 0
        batch_lr = optimizer.param_groups[0]['lr']
        for epoch in range(1, num_epochs + 1):
            tqdm_bar = tqdm(self.loader)
            batch_counter = 0
            total_loss = 0.0
            for idx, images, labels in tqdm_bar:
                # Move data to GPU
                images = images.cuda(0)
                labels = labels.cuda(0)
                optimizer.zero_grad()
            
                # Forward-prop, then calculate gradients and one step through optimizer
                feats = self.backbone.forward(images)
                logits = self.classifier.forward(feats)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()
            
                # Update progress bar, loss-tracking and other variables.
                total_loss += loss.item()
                batch_counter += 1
                total_batches += 1
                avg_loss = total_loss / batch_counter
                batch_lr = optimizer.param_groups[0]['lr']
                if self.config_dict['save']:
                    tb_writer.add_scalar(tag=comp_name + '/LR', scalar_value=batch_lr, global_step=total_batches)
                
                    if (total_batches % len(self.loader) == 1 or total_batches == num_epochs * len(self.loader)) and \
                            tr_args['log_test_error'] and self.test_loader is not None:
                    
                        test_loss_total = 0.0
                        batch_count_loss = 0
                        for _, (indices_test, images_test, labels_test) in enumerate(self.test_loader):
                            images_test = images_test.cuda(non_blocking=True)
                            labels_test = labels_test.cuda(non_blocking=True)
                            if len(images_test.size()) == 5:
                                b_size, n_crops, c, h, w = images_test.size()
                            else:
                                b_size, c, h, w = images_test.size()
                                n_crops = 1
                            with torch.no_grad():
                                feats_test = self.backbone.forward(images_test.view(-1, c, h, w))
                                logits_test = self.classifier(feats_test)
                                logits_test_avg = logits_test.view(b_size, n_crops, -1).mean(1)
                                test_loss = nn.CrossEntropyLoss()(logits_test_avg, labels_test)
                            test_loss_total += test_loss.item()
                            batch_count_loss += 1
                        avg_loss_test = test_loss_total / batch_count_loss
                        tb_writer.add_scalars(main_tag=comp_name + '/Loss',
                                              tag_scalar_dict={'avg_loss_train': avg_loss,
                                                               'batch_loss_train': loss.item(),
                                                               'loss_test': avg_loss_test
                                                               },
                                              global_step=total_batches)
                    else:
                        tb_writer.add_scalars(main_tag=comp_name + '/Loss',
                                              tag_scalar_dict={'avg_loss_train': avg_loss,
                                                               'batch_loss_train': loss.item()
                                                               },
                                              global_step=total_batches)
            
                tqdm_bar.set_description("Epoch: {:d}/{:d}  LR: {:.5f}  Loss: {:.4f}".format(epoch, num_epochs,
                                                                                             batch_lr, avg_loss))
            
                combined_scheduler.step()
        
            # Write average training error at end of epoch into log file.
            logger.info("Epoch: {:d}/{:d}  LR: {:.5f}  Loss: {:.4f}".format(epoch, num_epochs, batch_lr, avg_loss))
        
            tqdm_bar.close()
    
        # If models have to be saved, save both classifier and backbone in the
        # directory specified in 'save_dir' of config_dict.
        if self.config_dict['save']:
            backbone_file_name = self.config_dict['save_dir'] + self.components[self.run_id] + "_" + \
                                 self.config_dict['model']['name'] + "_backbone.pt"
            logger.info("Saving backbone to %s", backbone_file_name)
            torch.save(self.backbone.state_dict(), backbone_file_name)
            classifier_file_name = self.config_dict['save_dir'] + self.components[self.run_id] + "_" \
                                   + self.config_dict['model']['name'] + "_classifier.pt"
            logger.info("Saving classifier to %s", classifier_file_name)
            torch.save(self.classifier.state_dict(), classifier_file_name)
    
    def eval_model(self, loader, dataset_name):
        """
        This method calculates the accuracies on the provided
        dataloader for the current model defined by backbone
        and classifier.
        """
        top1acc = 0
        top5acc = 0
        tot_train_points = 0
        save_csv_file = None
        csv_writer = None
        logger.debug("Preparing to run eval on %s", dataset_name)
        if self.config_dict['save']:
            eval_num = 0
            for comp_idx in range(0, self.run_id):
                if "train" not in self.components[comp_idx]:
                    eval_num += 1
            pred_csv_file_name = "{}eval_{}_{}.csv".format(self.config_dict['save_dir'], str(eval_num), dataset_name)
            logger.debug("Saving predictions to %s", pred_csv_file_name)
            save_csv_file = open(pred_csv_file_name, "w")
            csv_writer = csv.writer(save_csv_file)
            csv_writer.writerow(["Index", "True Label", "Predicted Label"])
        
        # Iterate over batches and calculate top-1 and top-5 accuracies
        for _, (indices, images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            if len(images.size()) == 5:
                b_size, n_crops, c, h, w = images.size()
            else:
                b_size, c, h, w = images.size()
                n_crops = 1
            with torch.no_grad():
                feats = self.backbone.forward(images.view(-1, c, h, w))
                logits = self.classifier(feats)
                logits_avg = logits.view(b_size, n_crops, -1).mean(dim=1)
            top, pred = utils.calc_accuracy(logits_avg, labels, topk=(1, 5))
            top1acc += top[0].item() * pred.shape[0]
            top5acc += top[1].item() * pred.shape[0]
            tot_train_points += pred.shape[0]
            if self.config_dict['save']:
                indices, labels, pred = indices.cpu().numpy(), labels.cpu().numpy(), pred.cpu().numpy()
                for idx in range(pred.shape[0]):
                    csv_writer.writerow([indices[idx], labels[idx], pred[idx]])
        top1acc /= tot_train_points
        top5acc /= tot_train_points
        logger.debug("Completed eval on %s", dataset_name)
        if self.config_dict['save']:
            save_csv_file.close()
        return top1acc, top5acc
    
    def test_model(self):
        """
        This method runs the test/eval components of the experiment.
        All required eval datasets should already be loaded in
        test_loaders.
        """
        comp = self.config_dict['components'][self.run_id]
        test_sets = self.config_dict[comp]
        # Freeze all params, put dropout/batchnorm params in test_mode and
        # call eval_model() method.
        for params in self.backbone.parameters():
            params.requires_grad = False
        for params in self.classifier.parameters():
            params.requires_grad = False
        self.backbone.eval()
        self.classifier.eval()
        top1_accs, top5_accs = [], []
        
        for loader_idx, _ in enumerate(test_sets):
            top1, top5 = self.eval_model(loader=self.test_loader, dataset_name=self.test_loader_name)
            top1_accs.append(top1)
            top5_accs.append(top5)
            logger.info("Accuracy on test set: %s is %f", self.test_loader_name, top1)
            if loader_idx < len(test_sets) - 1:
                self.set_testing_data(idx_test=loader_idx + 1)
        return top1_accs, top5_accs
    
    def prepare_run(self):
        """
        This method prepares for the next component to be run.
        Looks at component name to determine if train or eval/test
        and loads data into the appropriate class variables
        """
        
        logger.info("Preparing for component %d: %s", self.run_id, self.components[self.run_id])
        component_name = self.components[self.run_id]
        if 'train' in component_name:
            # training component, load training data and training args.
            self.config_dict = parse_config.validate_toybox_views(yaml_dict=self.config_dict,
                                                                  target_component=component_name)
            logger.debug("Component %s is training component. Setting training data and loading run args.....",
                         component_name)
            if 'log_test_error' not in self.config_dict[component_name]['args']:
                self.config_dict[component_name]['args']['log_test_error'] = False
            self.set_training_data(comp=component_name)
            self.config_dict = parse_config.validate_lr_scheduler(yaml_dict=self.config_dict,
                                                                  target_component=component_name,
                                                                  num_batches=len(self.loader))
            self.run_args = self.config_dict[component_name]['args']
            if 'core50' in self.config_dict[component_name]['data'].lower():
                self.classifier = nn.Linear(self.fc_size, 10)
                self.classifier.cuda()
        else:
            # testing component, load testing data.
            logger.debug("Component %s is testing component. Loading testing data....", component_name)
            self.set_testing_data(idx_test=0)
        logger.info("Ready to run component %d: %s....", self.run_id, component_name)
    
    def run(self):
        """
        Run the next component in list of components.
        Uses component name to determine if train or
        eval method and executes methods accordingly.
        """
        
        component = self.components[self.run_id]
        if 'train' in component:
            self.train_model()
        else:
            top1_accs, _ = self.test_model()
            logger.info("Accuracies on test set: {}".format(", ".join(map(str, top1_accs))))
        self.run_id += 1
    
    def set_training_data(self, comp):
        """
        Set the training data for a training component to be run.
        """
        # Check if training dataset is valid
        train_dataset_name = self.config_dict[comp]['data'].lower()
        tr_args = self.config_dict[comp]['args']
        
        num_matches = 0
        for dname in ALL_DATASETS:
            if dname.lower() in train_dataset_name:
                num_matches += 1
        assert num_matches == 1, "Make sure the dataset name for component {} is unique and one of [{}]..." \
                                 "".format(comp, ", ".join(ALL_DATASETS))
        
        if 'toybox' in train_dataset_name:
            logger.debug("The training dataset is Toybox. Preparing to initialize train_data....")
            train_transform = self.get_train_transform(mean=TOYBOX_MEAN, std=TOYBOX_STD)
            train_data = parse_config.get_dataset(dataset_toybox, 'ToyboxDataset',
                                                  root=TOYBOX_DATA_PATH,
                                                  transform=train_transform, num_instances=tr_args['num_instances'],
                                                  rng=np.random.default_rng(self.config_dict['seed']), hypertune=False,
                                                  num_images_per_class=tr_args['num_images_per_class'],
                                                  views=tr_args['views'], train=True)
            if self.config_dict[comp]['args']['log_test_error']:
                logger.debug("Setting testing data for logging test error during training....")
                self.set_testing_data(test_set='toybox_test')
        elif "in12" in train_dataset_name:
            logger.debug("The training dataset is ImageNet+COCO. Preparing to initialize train_data....")
            train_transform = self.get_train_transform(mean=IN12_MEAN, std=IN12_STD)
            train_data = parse_config.get_dataset(dataset_imagenet12, 'DataLoaderGeneric', root=IN12_DATA_PATH,
                                                  transform=train_transform, train=True)
            if self.config_dict[comp]['args']['log_test_error']:
                self.set_testing_data(test_set='imagenet_test')
        else:
            logger.debug("The training dataset is CoRE50. Preparing to initialize train_data....")
            train_transform = self.get_train_transform(mean=CORE50_MEAN, std=CORE50_STD)
            train_data = parse_config.get_dataset(dataset_core50, 'DatasetCoRE50',
                                                  root=CORE50_DATA_PATH, train=True,
                                                  transform=train_transform, fraction=0.1)
            if self.config_dict[comp]['args']['log_test_error']:
                self.set_testing_data(test_set='core50_test')
        
        logger.debug("Initialized train_data. Preparing to initialize dataloader....")
        self.loader = torch.utils.data.DataLoader(train_data, batch_size=self.global_params['batch_size'],
                                                  shuffle=True, num_workers=self.global_params['num_workers'])
        logger.debug("Initialized dataloader....")
    
    def set_testing_data(self, idx_test=None, test_set=None, ten_crop=False):
        """
        Prepare dataloaders for testing component to be run.
        """
        assert idx_test is None or test_set is None, "One of idx_test and test_set must be None...."
        comp = self.config_dict['components'][self.run_id]

        if test_set is None:
            test_set = self.config_dict[comp][idx_test]
        else:
            ten_crop = False
        logger.info("Preparing to load dataloader for {} for evaluation....".format(test_set))
        if 'toybox' in test_set:
            test_transform = self.get_test_transform(mean=TOYBOX_MEAN, std=TOYBOX_STD, ten_crop=ten_crop)
            
            if 'train' in test_set:
                self.test_loader_name = "toybox_train"
                logger.debug("Loading toybox train set....")
                test_data = parse_config.get_dataset(dataset_toybox, 'ToyboxDataset',
                                                     root=TOYBOX_DATA_PATH,
                                                     transform=test_transform, num_instances=-1,
                                                     rng=np.random.default_rng(0), hypertune=False,
                                                     num_images_per_class=1000, train=True)
            else:
                self.test_loader_name = "toybox_test"
                logger.debug("Loading toybox test set....")
                test_data = parse_config.get_dataset(dataset_toybox, 'ToyboxDataset',
                                                     root=TOYBOX_DATA_PATH,
                                                     transform=test_transform, rng=np.random.default_rng(0),
                                                     hypertune=False, train=False)
        elif "in12" in test_set:
            test_transform = self.get_test_transform(mean=IN12_MEAN, std=IN12_STD, ten_crop=ten_crop)
            
            if 'train' in test_set:
                self.test_loader_name = "imagenet_coco_train"
                logger.debug("Loading imagenet+coco train set....")
                test_data = parse_config.get_dataset(dataset_imagenet12, 'DataLoaderGeneric', train=True,
                                                     root=IN12_DATA_PATH, transform=test_transform)
            else:
                self.test_loader_name = "imagenet_coco_test"
                logger.debug("Loading imagenet+coco test set....")
                test_data = parse_config.get_dataset(dataset_imagenet12, 'DataLoaderGeneric', train=False,
                                                     root=IN12_DATA_PATH, transform=test_transform)
        else:
            test_transform = self.get_test_transform(mean=CORE50_MEAN, std=CORE50_STD, ten_crop=ten_crop)
    
            if 'train' in test_set:
                self.test_loader_name = "core50_train"
                logger.debug("Loading core50 train set....")
                test_data = parse_config.get_dataset(dataset_core50, 'DatasetCoRE50', train=True,
                                                     root=CORE50_DATA_PATH,
                                                     transform=test_transform, fraction=0.1)
            else:
                self.test_loader_name = "core50_test"
                logger.debug("Loading core50 test set....")
                test_data = parse_config.get_dataset(dataset_core50, 'DatasetCoRE50',
                                                     root=CORE50_DATA_PATH, train=False,
                                                     transform=test_transform)
        
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                                       shuffle=True, num_workers=self.global_params['num_workers'])
        
        logger.info("Test set %s for component %s loaded and ready to run.", test_set, comp)
    
    def run_components(self):
        """Method to run all components in sequence"""
        for run_idx in range(len(self.components)):
            # Needed prep for first component already done by __init__()
            self.run()
            if run_idx < len(self.components) - 1:
                self.prepare_run()


if __name__ == "__main__":
    args = vars(utils.get_parser())
    
    log_level = getattr(logging, args['log_level'].upper())
    logging.basicConfig(format=LOG_FORMAT_TERMINAL, level=log_level)
    logger = logging.getLogger()
    
    logger.info("Initialized logger. Trying to load config file: %s", args['config_file'])
    try:
        assert os.path.isfile(CONFIG_DIR + args['config_file']), "File not found: " + args['config_file']
    except AssertionError:
        logger.exception("File %s does not exist", args['config_file'])
        raise
    f_stream = open(CONFIG_DIR + args['config_file'], 'r')
    yaml_dict = yaml.safe_load(f_stream)
    for key, val in yaml_dict.items():
        logger.debug(key + " : " + str(val))
    
    logger.info("Loaded config file.... Initializing Experiment....")
    yaml_dict['seed'] = args['seed']
    if 'save' in yaml_dict.keys():
        if yaml_dict['save']:
            dt_now = datetime.datetime.now()
            
            tb_writer = tb.SummaryWriter(log_dir="../runs/" + dt_now.strftime("%b-%d-%Y-%H-%M") + "/")
            dir_name = OUTPUT_DIR + dt_now.strftime("%b-%d-%Y-%H-%M") + "/"
            
            logger.info("Creating directory to save into: %s, and adding dir to config file....", dir_name)
            os.makedirs(dir_name, exist_ok=False)
            yaml_dict['save_dir'] = dir_name
            logging_file = yaml_dict['save_dir'] + "log.txt"
            logfile_handler = logging.FileHandler(logging_file)
            logging_formatter = logging.Formatter(LOG_FORMAT_FILE)
            logfile_handler.setFormatter(logging_formatter)
            logger.addHandler(logfile_handler)
    else:
        yaml_dict['save'] = False
        
    exp = Experiment(config_dict=yaml_dict)
    exp.run_components()
    
    if yaml_dict['save']:
        with open(yaml_dict['save_dir'] + "exp_config.yaml", "w") as out_config_file:
            yaml.dump(yaml_dict, out_config_file, default_flow_style=False)
