"""
This file contains code for performing supervised learning experiments on IN-12
"""
import torch
import torch.utils.data as torchdata
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import tqdm
import torch.utils.tensorboard as tb
import logging
import datetime
import os

import dataset_imagenet12
import utils

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)
TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)

ALL_DATASETS = ["Toybox", "IN12"]
TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"

OUTPUT_DIR = "../out/IN12_SUP/"
RUNS_DIR = "../runs/IN12_SUP/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

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


class Network(nn.Module):
    """
    Network for supervised experiments
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False, num_classes=1000)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.fc_size, 12)
    
    def forward(self, x):
        """
        Forward method for network
        """
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


class NNTrainer:
    """
    Train the model and run eval periodically with specified data
    """
    
    def __init__(self, fraction, epochs, logr, hypertune=True, save=False, log_test_error=False):
        self.fraction = fraction
        self.hypertune = hypertune
        self.epochs = epochs
        self.save = save
        self.logger = logr
        self.log_test_error = log_test_error
        
        self.net = Network()
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
        ])
        
        self.dataset_train = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, fraction=self.fraction,
                                                                  train=True, hypertune=self.hypertune,
                                                                  transform=self.train_transform)
        self.dataset_test = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False,
                                                                 hypertune=self.hypertune,
                                                                 transform=self.train_transform)
        
        self.train_loader = torchdata.DataLoader(self.dataset_train, batch_size=128, num_workers=4, shuffle=True,
                                                 persistent_workers=True, pin_memory=True)
        
        self.test_loader = torchdata.DataLoader(self.dataset_test, batch_size=128, num_workers=4, shuffle=True,
                                                persistent_workers=True, pin_memory=True)

    def prepare_model_for_run(self):
        """
        This method prepares the backbone and classifier for a training run.
        """
        
        self.net.classifier.apply(utils.weights_init)
        # Set which weights will be updated according to config.
        # Set train() for network components. Only matters for regularization
        # techniques like dropout and batchnorm.
        # Set optimizer and add which weights are to be optimized acc. to config.
        for params in self.net.classifier.parameters():
            params.requires_grad = True
        for params in self.net.backbone.parameters():
            params.requires_grad = False
    
        self.net.backbone.eval()
        self.net.classifier.train()
        optimizer = torch.optim.SGD(self.net.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        
        return optimizer
    
    def train(self):
        """
        Train the model
        """
        
        optimizer = self.prepare_model_for_run()
        self.net.backbone = self.net.backbone.cuda()
        self.net.classifier = self.net.classifier.cuda()
        if self.epochs == 0:
            # No training required if number of epochs is 0
            return
        
        total_params = sum(p.numel() for p in self.net.backbone.parameters())
        train_params = sum(p.numel() for p in self.net.backbone.parameters() if p.requires_grad)
        self.logger.info("{}/{} parameters in the backbone are trainable...".format(train_params, total_params))
        total_params = sum(p.numel() for p in self.net.classifier.parameters())
        train_params = sum(p.numel() for p in self.net.classifier.parameters() if p.requires_grad)
        self.logger.info("{}/{} parameters in the classifier are trainable...".format(train_params, total_params))
        # Set lr scheduler for training experiment.
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                                             total_iters=2 * len(self.train_loader) - 1)
        
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                                   schedulers=[warmup_scheduler, scheduler],
                                                                   milestones=[2 * len(self.train_loader) - 1])
        
        num_epochs = self.epochs
        avg_loss = 0.0
        total_batches = 0
        batch_lr = optimizer.param_groups[0]['lr']
        dt_now = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M")
        tb_writer = tb.SummaryWriter(log_dir=RUNS_DIR + dt_now + "/")
        for epoch in range(1, num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.train_loader)
            batch_counter = 0
            total_loss = 0.0
            for idx, images, labels in tqdm_bar:
                # Move data to GPU
                images = images.cuda(0)
                labels = labels.cuda(0)
                optimizer.zero_grad()
                
                # Forward-prop, then calculate gradients and one step through optimizer
                feats = self.net.backbone.forward(images)
                logits = self.net.classifier.forward(feats)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()
                
                # Update progress bar, loss-tracking and other variables.
                total_loss += loss.item()
                batch_counter += 1
                total_batches += 1
                avg_loss = total_loss / batch_counter
                batch_lr = optimizer.param_groups[0]['lr']
                if self.save:
                    tb_writer.add_scalar(tag='Train/LR', scalar_value=batch_lr, global_step=total_batches)
                    
                    if (total_batches % len(self.train_loader) == 1 or
                        total_batches == num_epochs * len(self.train_loader)) and \
                            self.log_test_error and \
                            self.test_loader is not None:
                        
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
                                feats_test = self.net.backbone.forward(images_test.view(-1, c, h, w))
                                logits_test = self.net.classifier(feats_test)
                                logits_test_avg = logits_test.view(b_size, n_crops, -1).mean(1)
                                test_loss = nn.CrossEntropyLoss()(logits_test_avg, labels_test)
                            test_loss_total += test_loss.item()
                            batch_count_loss += 1
                        avg_loss_test = test_loss_total / batch_count_loss
                        tb_writer.add_scalars(main_tag='/Loss',
                                              tag_scalar_dict={'avg_loss_train': avg_loss,
                                                               'batch_loss_train': loss.item(),
                                                               'loss_test': avg_loss_test
                                                               },
                                              global_step=total_batches)
                    else:
                        tb_writer.add_scalars(main_tag='/Loss',
                                              tag_scalar_dict={'avg_loss_train': avg_loss,
                                                               'batch_loss_train': loss.item()
                                                               },
                                              global_step=total_batches)
                
                tqdm_bar.set_description("Epoch: {:d}/{:d}  LR: {:.5f}  Loss: {:.4f}".format(epoch, num_epochs,
                                                                                             batch_lr, avg_loss))
                
                combined_scheduler.step()
            
            # Write average training error at end of epoch into log file.
            self.logger.info("Epoch: {:d}/{:d}  LR: {:.5f}  Loss: {:.4f}".format(epoch, num_epochs, batch_lr, avg_loss))
            
            tqdm_bar.close()
        
        # If models have to be saved, save both classifier and backbone in the
        # directory specified in 'save_dir' of config_dict.
        if self.save:
            os.makedirs(OUTPUT_DIR + dt_now, exist_ok=True)
            backbone_file_name = OUTPUT_DIR + dt_now + "/backbone_final.pt"
            self.logger.info("Saving backbone to %s", backbone_file_name)
            torch.save(self.net.backbone.state_dict(), backbone_file_name)
            classifier_file_name = OUTPUT_DIR + dt_now + "/classifier_final.pt"
            self.logger.info("Saving classifier to %s", classifier_file_name)
            torch.save(self.net.classifier.state_dict(), classifier_file_name)


if __name__ == "__main__":
    log_level = getattr(logging, "INFO")
    logging.basicConfig(format=LOG_FORMAT_TERMINAL, level=log_level)
    logger = logging.getLogger()
    exp = NNTrainer(fraction=0.1, epochs=3, logr=logger, hypertune=True, save=True, log_test_error=False)
    exp.train()