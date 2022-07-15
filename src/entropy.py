"""
Module that implements method that learns from entropy on target distribution
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import tqdm
import torch.utils.tensorboard as tb
import argparse

import dataset_imagenet12
import dataset_toybox
import utils

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)
TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)


class EntModel:
    """
        Model that learns from entropy on target distribution
        """
    def __init__(self, model_path, exp_args):
        self.args = exp_args
        self.model_path = model_path
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(self.fc_size, 12)
        self.softmax = nn.Softmax(dim=1)
        
        backbone_path = self.model_path + "backbone_trainer_resnet18_backbone.pt"
        classifier_path = self.model_path + "backbone_trainer_resnet18_classifier.pt"
        
        self.backbone.load_state_dict(torch.load(backbone_path))
        # self.fc.load_state_dict(torch.load(classifier_path))
        
        self.backbone.eval()
        self.backbone.cuda()
        self.fc.eval()
        self.fc.cuda()
    
    def train_model(self, data, eval_data, ent_train_data):
        """
        Train loaded model with data provided
        """
        tb_writer = tb.SummaryWriter(log_dir="../runs/entmod/")
        data_loader = torchdata.DataLoader(data, batch_size=self.args['batch_size'], num_workers=4, shuffle=True,
                                           pin_memory=True, persistent_workers=True)
        ent_data_loader = torchdata.DataLoader(ent_train_data, batch_size=self.args['batch_size'], num_workers=4,
                                               shuffle=True)
        ent_loader_iter = iter(ent_data_loader)
        optimizer = torch.optim.SGD(self.fc.parameters(), lr=self.args['lr'], weight_decay=1e-6)
        for params in self.fc.parameters():
            params.requires_grad = True
        if self.args['backbone']:
            for params in self.backbone.parameters():
                params.requires_grad = True
            optimizer.add_param_group({'params': self.backbone.parameters()})
            
        num_epochs = self.args['epochs']
        entropy = self.run_dataset(eval_data)
        self.fc.train()
        if self.args['backbone']:
            self.backbone.train()
        total_batches = 0
        # tb_writer.add_scalar(tag='Entropy', scalar_value=entropy, global_step=total_batches)
        for ep in range(num_epochs):
            num_batches = 0
            total_loss = 0
            tqdm_bar = tqdm.tqdm(data_loader)
            for idx, images, labels in tqdm_bar:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                feats = self.backbone.forward(images)
                logits = self.fc.forward(feats)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
                try:
                    _, images_ent, _ = next(ent_loader_iter)
                except StopIteration:
                    ent_loader_iter = iter(ent_data_loader)
                    _, images_ent, _ = next(ent_loader_iter)
            
                images_ent = images_ent.cuda()
                feats_ent = self.backbone.forward(images_ent)
                activations_ent = self.fc.forward(feats_ent)
                train_entropy = torch.mean(self.calc_entropy(activations_ent))
                combined_loss = loss + self.args['lambda'] * train_entropy
                num_batches += 1
                total_loss += loss.item()
                total_batches += 1
                tqdm_bar.set_description("Epoch: {}/{}  LR: {:.4f}  CE Loss: {:.4f} Train Entropy: {:.4f},  "
                                         "Total Loss: {:.4f}  Test Entropy: {:.6f}".
                                         format(ep+1, num_epochs, optimizer.param_groups[0]['lr'],
                                                total_loss/num_batches, train_entropy.item(), combined_loss.item(),
                                                entropy))
            
                combined_loss.backward()
                optimizer.step()
                # if (num_batches+1) % 20 == 0:
                #     entropy = self.run_dataset(eval_data)
                #     self.fc.train()
                tb_writer.add_scalar(tag='Entropy', scalar_value=entropy, global_step=total_batches)
                tb_writer.add_scalar(tag='Loss/Toybox_Train', scalar_value=total_loss/num_batches,
                                     global_step=total_batches)
        
            entropy = self.run_dataset(eval_data)
            tb_writer.add_scalar(tag='Entropy', scalar_value=entropy, global_step=total_batches)
            print("Average Entropy: {}".format(entropy))
        
            test_loss = self.get_loss(eval_data)
            tb_writer.add_scalar(tag='Loss/IN12_Test', scalar_value=test_loss, global_step=total_batches)
            print("Test Loss: {}".format(test_loss))
        
            self.get_acc(eval_data)
        
            self.fc.train()
            if self.args['backbone']:
                self.backbone.train()
        tb_writer.close()
            
    def calc_entropy(self, activations):
        """
        Calculate entropy based on
        """
        logits = self.softmax(activations)
        log_logits = torch.log2(logits)
        entropy = logits * log_logits * -1
        return entropy
    
    def run_dataset(self, data):
        """
        Pass dataset through network and calculate average entropy
        """
        data_loader = torchdata.DataLoader(data, batch_size=64, num_workers=4, pin_memory=False,
                                           persistent_workers=True)
        total_entropy = 0.0
        batches = 0
        self.backbone.eval()
        self.fc.eval()
        for _, images, _ in data_loader:
            images = images.cuda()
            with torch.no_grad():
                feats = self.backbone(images)
                activations = self.fc(feats)
                batch_entropy = self.calc_entropy(activations)
                total_entropy += torch.mean(batch_entropy)
                batches += 1
        
        return total_entropy/batches

    def get_loss(self, data):
        """
        Pass dataset through network and calculate loss
        """
        data_loader = torchdata.DataLoader(data, batch_size=64, num_workers=4, pin_memory=False,
                                           persistent_workers=True)
        
        batches = 0
        total_loss = 0.0
        self.backbone.eval()
        self.fc.eval()
        for _, images, labels in data_loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                feats = self.backbone(images)
                activations = self.fc(feats)
                loss = nn.CrossEntropyLoss()(activations, labels)
                total_loss += loss.item()
                batches += 1
    
        return total_loss / batches

    def get_acc(self, data):
        """
        Pass dataset through network and calculate accuracy
        """
        data_loader = torchdata.DataLoader(data, batch_size=64, num_workers=4, pin_memory=False,
                                           persistent_workers=True)
    
        self.backbone.eval()
        self.fc.eval()
        top1acc = 0.0
        tot_train_points = 0
        for _, images, labels in data_loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                feats = self.backbone(images)
                activations = self.fc(feats)
                top, pred = utils.calc_accuracy(activations, labels, topk=(1, ))
                top1acc += top[0].item() * pred.shape[0]
                tot_train_points += pred.shape[0]
        top1acc /= tot_train_points
        print("Accuracy on test set: {}".format(top1acc))
    
        return top1acc
                
        
def get_parser():
    """
    Create parser and return arguments for experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", "-l", default=1, type=int)
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--backbone", "-bb", default=False, action='store_true')
    parser.add_argument("--lr", "-lr", default=0.1, type=float)
    parser.add_argument("--batch-size", "-b", default=64, type=int)
    
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = vars(get_parser())
    model = EntModel(model_path="../out/toybox_baseline/", exp_args=args)
    transform_in12 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                         transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
    in12_test_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False, transform=transform_in12)

    in12_train_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True, transform=transform_in12)
    
    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                           transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)])
    toybox_train_data = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, rng=np.random.default_rng(), train=True,
                                                     num_instances=10, num_images_per_class=2000,
                                                     transform=transform_toybox)
    model.train_model(data=toybox_train_data, eval_data=in12_test_data, ent_train_data=in12_train_data)
    