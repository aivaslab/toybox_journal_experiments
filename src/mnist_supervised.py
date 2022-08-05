"""
Code for supervised experiments on MNIST dataset
"""
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.nn as nn
import tqdm
import torch.utils.tensorboard as tb
import logging
import datetime
import os
import csv
import argparse
import pickle

import utils
import mean_teacher
import dataset_mnist50

OUTPUT_DIR = "../out/MNIST_SUP/"
RUNS_DIR = "../runs/MNIST_SUP/"
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

MNIST_MEAN = (0.1309, 0.1309, 0.1309)
MNIST_STD = (0.2893, 0.2893, 0.2893)


class Network(nn.Module):
    """
    Network for supervised experiments
    """
    
    def __init__(self, network_file_name):
        super().__init__()
        self.network_file_name = network_file_name
        self.network = mean_teacher.Network(n_classes=10)
        if os.path.isfile(self.network_file_name):
            print("Loading backbone from {}".format(self.network_file_name))
            self.network.load_state_dict(torch.load(self.network_file_name))
    
    def forward(self, x):
        """
        Forward method for network
        """
        logits = self.network.forward(x)
        return logits


def get_transform(idx):
    """Return the train_transform"""
    tr = transforms.Compose([transforms.ToPILImage(),
                             transforms.Grayscale(3),
                             ])
    if idx > 0:
        tr.transforms.append(transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)))
    if idx > 1:
        tr.transforms.append(transforms.RandomInvert(p=0.5))
    if idx > 2:
        tr.transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    if idx > 3:
        tr.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4))
    
    tr.transforms.append(transforms.Resize(32))
    tr.transforms.append(transforms.ToTensor())
    tr.transforms.append(transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD))
    
    return tr

    
class NNTrainer:
    """
    Train the model and run eval periodically with specified data
    """
    
    def __init__(self, network_file, fraction, epochs, logr, hypertune=True, save=False, log_test_error=False,
                 save_frequency=100, save_frequency_batch=10000, lr=0.01, transform=4):
        self.network_file = network_file
        self.fraction = fraction
        self.hypertune = hypertune
        self.epochs = epochs
        self.save = save
        self.logger = logr
        self.lr = lr
        self.log_test_error = log_test_error
        self.save_frequency = save_frequency
        self.save_frequency_batch = save_frequency_batch
        self.transform = transform
        
        self.net = Network(network_file_name=network_file)
        
        self.train_transform = get_transform(idx=self.transform)
        self.test_transform = get_transform(idx=0)
        print(self.train_transform)
        print(self.test_transform)
        self.dataset_train = dataset_mnist50.DatasetMNIST50(root="../data/", train=True, transform=self.train_transform)
        self.dataset_test = dataset_mnist50.DatasetMNIST50(root="../data/", train=False, transform=self.test_transform)
        
        self.train_loader = torchdata.DataLoader(self.dataset_train, batch_size=256, num_workers=4, shuffle=True,
                                                 persistent_workers=True, pin_memory=True)
        self.test_loader = torchdata.DataLoader(self.dataset_test, batch_size=128, num_workers=4, shuffle=False,
                                                persistent_workers=True, pin_memory=True)
    
    def prepare_model_for_run(self):
        """
        This method prepares the backbone and classifier for a training run.
        """
        
        self.net.network.apply(utils.weights_init)
        # Set which weights will be updated according to config.
        # Set train() for network components. Only matters for regularization
        # techniques like dropout and batchnorm.
        # Set optimizer and add which weights are to be optimized acc. to config.
        for params in self.net.network.parameters():
            params.requires_grad = True
        for params in self.net.network.fc4.parameters():
            params.requires_grad = True
            
        self.net.network.train()
        optimizer = torch.optim.SGD(self.net.network.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        
        return optimizer
    
    def train(self):
        """
        Train the model
        """
        self.net.network = self.net.network.cuda()
        if self.epochs == 0:
            # No training required if number of epochs is 0
            return
        optimizer = self.prepare_model_for_run()
        
        total_params = sum(p.numel() for p in self.net.network.parameters())
        train_params = sum(p.numel() for p in self.net.network.parameters() if p.requires_grad)
        self.logger.info("{}/{} parameters in the backbone are trainable...".format(train_params, total_params))
        
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
        batch_accs = {}
        epoch_accs = {}
        for epoch in range(1, num_epochs + 1):
            tqdm_bar = tqdm.tqdm(self.train_loader)
            batch_counter = 0
            total_loss = 0.0
            for idxs, images, labels in tqdm_bar:
                # Move data to GPU
                images = images.cuda(0)
                labels = labels.cuda(0)
                optimizer.zero_grad()
                
                # Forward-prop, then calculate gradients and one step through optimizer
                logits = self.net.network.forward(images)
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
                    tb_writer.add_scalars(main_tag='Train/Loss',
                                          tag_scalar_dict={'avg_loss_train': avg_loss,
                                                           'batch_loss_train': loss.item()
                                                           },
                                          global_step=total_batches)
                
                tqdm_bar.set_description("Epoch: {:d}/{:d}  LR: {:.5f}  Loss: {:.4f}".format(epoch, num_epochs,
                                                                                             batch_lr, avg_loss))
                
                combined_scheduler.step()
                
                if self.save and total_batches % self.save_frequency_batch == 0:
                    os.makedirs(OUTPUT_DIR + dt_now, exist_ok=True)
                    network_file_name = OUTPUT_DIR + dt_now + "/mnist_network_batch_" + str(total_batches) + ".pt"
                    self.logger.info("Saving network to %s", network_file_name)
                    torch.save(self.net.network.state_dict(), network_file_name)
                    acc = self.eval_model(csv_file_name=OUTPUT_DIR + dt_now + "/mnist_eval_batch_" + str(total_batches)
                                                                   + ".csv")
                    batch_accs[total_batches] = acc
            
            # Write average training error at end of epoch into log file.
            self.logger.info("Epoch: {:d}/{:d}  LR: {:.5f}  Loss: {:.4f}".format(epoch, num_epochs, batch_lr, avg_loss))
            
            tqdm_bar.close()
            
            if self.save and epoch % self.save_frequency == 0:
                os.makedirs(OUTPUT_DIR + dt_now, exist_ok=True)
                network_file_name = OUTPUT_DIR + dt_now + "/mnist_network_epoch_" + str(epoch) + ".pt"
                self.logger.info("Saving network to %s", network_file_name)
                torch.save(self.net.network.state_dict(), network_file_name)
                acc = self.eval_model(csv_file_name=OUTPUT_DIR + dt_now + "/mnist_eval_epoch_" + str(epoch) + ".csv")
                epoch_accs[epoch] = acc
        
        # If models have to be saved, save both classifier and backbone in the
        # directory specified in 'save_dir' of config_dict.
        if self.save:
            os.makedirs(OUTPUT_DIR + dt_now, exist_ok=True)
            network_file_name = OUTPUT_DIR + dt_now + "/mnist_network_final.pt"
            self.logger.info("Saving network to %s", network_file_name)
            torch.save(self.net.network.state_dict(), network_file_name)
            
            acc = self.eval_model(csv_file_name=OUTPUT_DIR + dt_now + "/mnist_eval_final.csv")
            batch_acc_file_name = OUTPUT_DIR + dt_now + "/batch_accs.pkl"
            batch_acc_file = open(batch_acc_file_name, "wb")
            pickle.dump(batch_accs, batch_acc_file, protocol=pickle.DEFAULT_PROTOCOL)
            batch_acc_file.close()

            epoch_acc_file_name = OUTPUT_DIR + dt_now + "/epoch_accs.pkl"
            epoch_acc_file = open(epoch_acc_file_name, "wb")
            pickle.dump(epoch_accs, epoch_acc_file, protocol=pickle.DEFAULT_PROTOCOL)
            epoch_acc_file.close()
        else:
            acc = self.eval_model()
        train_acc = self.eval_model(training=True)
        self.logger.info("Final accuracy on train set: {:.2f}".format(train_acc))
        self.logger.info("Final accuracy on test set: {:.2f}".format(acc))
        
        print(batch_accs)
        print(epoch_accs)
    
    def eval_model(self, csv_file_name=None, training=False):
        """
        This method calculates the accuracies on the provided
        dataloader for the current model defined by backbone
        and classifier.
        """
        self.net.network.eval()
        top1acc = 0
        tot_train_points = 0
        
        save_csv_file = None
        csv_writer = None
        if csv_file_name is not None:
            self.logger.debug("Saving predictions to %s", csv_file_name)
            save_csv_file = open(csv_file_name, "w")
            csv_writer = csv.writer(save_csv_file)
            csv_writer.writerow(["Index", "True Label", "Predicted Label"])
        if training:
            loader = self.train_loader
        else:
            loader = self.test_loader
        # Iterate over batches and calculate top-1 accuracy
        for _, (indices,  images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            if len(images.size()) == 5:
                b_size, n_crops, c, h, w = images.size()
            else:
                b_size, c, h, w = images.size()
                n_crops = 1
            with torch.no_grad():
                logits = self.net.network.forward(images)
                logits_avg = logits.view(b_size, n_crops, -1).mean(dim=1)
            top, pred = utils.calc_accuracy(logits_avg, labels, topk=(1,))
            top1acc += top[0].item() * pred.shape[0]
            tot_train_points += pred.shape[0]
            if csv_file_name is not None:
                indices, labels, pred = indices.cpu().numpy(), labels.cpu().numpy(), pred.cpu().numpy()
                for idx in range(pred.shape[0]):
                    csv_writer.writerow([indices[idx], labels[idx], pred[idx]])
        top1acc /= tot_train_points
        if csv_file_name is not None:
            save_csv_file.close()
        self.net.network.train()
        return top1acc


class Experiment:
    """
    Experiment class
    """
    
    def __init__(self, exp_args):
        self.exp_args = exp_args
        print(self.exp_args)
        log_level = getattr(logging, self.exp_args['log_level'].upper())
        logging.basicConfig(format=LOG_FORMAT_TERMINAL, level=log_level)
        logger = logging.getLogger()
        self.trainer = NNTrainer(network_file=self.exp_args['backbone'],
                                 fraction=self.exp_args['fraction'],
                                 epochs=self.exp_args['epochs'],
                                 lr=self.exp_args['lr'],
                                 logr=logger,
                                 hypertune=not self.exp_args['final'],
                                 transform=self.exp_args['transform'],
                                 save=self.exp_args['save'],
                                 log_test_error=self.exp_args['log_test_error'],
                                 save_frequency=self.exp_args['save_frequency'],
                                 save_frequency_batch=self.exp_args['save_frequency_batch']
                                 )
    
    def run(self):
        """
        Run the experiment
        """
        self.trainer.train()
        acc = self.trainer.eval_model()
        print("Final test accuracy:{:.2f}".format(acc))


def get_parser():
    """Generate parser for the experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--backbone", "-bb", default="", type=str)
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--fraction", "-f", default=0.5, type=float)
    parser.add_argument("--lr", "-lr", default=0.1, type=float)
    parser.add_argument("--save", "-s", default=False, action='store_true')
    parser.add_argument("--log-test-error", "-lte", default=False, action='store_true')
    parser.add_argument("--final", default=False, action='store_true')
    parser.add_argument("--log-level", "-ll", default="info", type=str)
    parser.add_argument("--save-frequency", "-sf", default=100, type=int)
    parser.add_argument("--save-frequency-batch", "-sfb", default=10000, type=int)
    parser.add_argument("--transform", "-t", default=5, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = vars(get_parser())
    
    exp = Experiment(exp_args=args)
    exp.run()
