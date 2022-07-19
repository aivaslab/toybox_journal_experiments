"""
Module that implements method that learns from entropy on target distribution
"""
import os

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
        log_dir = "../runs/" + self.args['log_dir']
        assert not os.path.isdir(log_dir), "Directory {} exists...".format(log_dir)
        model_out_dir = "../out/" + self.args['log_dir']
        os.makedirs(model_out_dir, exist_ok=False)
        tb_writer = tb.SummaryWriter(log_dir=log_dir)
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
            
        self.fc.train()
        if self.args['backbone']:
            self.backbone.train()
        
        entropy = self.run_dataset(eval_data)
        tb_writer.add_scalar(tag='IN12 Test Entropy', scalar_value=entropy, global_step=0)
        test_loss = self.get_loss(eval_data)
        tb_writer.add_scalar(tag='IN12 Test Loss', scalar_value=test_loss, global_step=0)

        total_batches = 0
        num_epochs = self.args['epochs']
        
        for ep in range(num_epochs):
            num_batches = 0
            total_loss = 0
            tqdm_bar = tqdm.tqdm(data_loader, ncols=155)
            for idx, images, labels in tqdm_bar:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                feats = self.backbone.forward(images)
                with torch.no_grad():
                    feat_norm = torch.mean(torch.linalg.vector_norm(feats, dim=1))
                logits = self.fc.forward(feats)
                # print(logits)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                if self.args['lambda'] > 0:
                    try:
                        _, images_ent, _ = next(ent_loader_iter)
                    except StopIteration:
                        ent_loader_iter = iter(ent_data_loader)
                        _, images_ent, _ = next(ent_loader_iter)
                
                    images_ent = images_ent.cuda()
                    feats_ent = self.backbone.forward(images_ent)
                    with torch.no_grad():
                        feat_ent_norm = torch.mean(torch.linalg.vector_norm(feats_ent, dim=1))
                    activations_ent = self.fc.forward(feats_ent)
                    train_entropy = torch.mean(self.calc_entropy(activations_ent))
                    combined_loss = loss + self.args['lambda'] * train_entropy
                else:
                    combined_loss = loss
                    train_entropy = torch.Tensor([0.0])
                    feat_ent_norm = 0.0
                num_batches += 1
                total_loss += loss.item()
                total_batches += 1
                tqdm_bar.set_description("Epoch: {}/{}  LR: {:.4f}  CE Loss: {:.3f}  Tr. Ent: {:.3f}  "
                                         "Total: {:.3f}  TB Norm: {:.2f}  IN12 Norm: {:.2f}".
                                         format(ep+1, num_epochs, optimizer.param_groups[0]['lr'],
                                                loss.item(), train_entropy.item(), combined_loss.item(),
                                                feat_norm, feat_ent_norm))
            
                combined_loss.backward()
                optimizer.step()
                
                tb_writer.add_scalar(tag='Loss/IN12_Entropy', scalar_value=train_entropy, global_step=total_batches)
                tb_writer.add_scalar(tag='Loss/TB_CrossEnt', scalar_value=loss.item(), global_step=total_batches)
                tb_writer.add_scalar(tag='Loss/Total', scalar_value=combined_loss.item(), global_step=total_batches)
                tb_writer.add_scalar(tag='Norm/Toybox', scalar_value=feat_norm, global_step=total_batches)
                tb_writer.add_scalar(tag='Norm/IN12', scalar_value=feat_ent_norm, global_step=total_batches)

            entropy = self.run_dataset(eval_data)
            tb_writer.add_scalar(tag='IN12 Test Entropy', scalar_value=entropy, global_step=total_batches)
            print("Average Entropy: {}".format(entropy))
        
            test_loss = self.get_loss(eval_data)
            tb_writer.add_scalar(tag='IN12 Test Loss', scalar_value=test_loss, global_step=total_batches)
            print("Test Loss: {}".format(test_loss))

            if (ep + 1) % 10 == 0:
                optimizer.param_groups[0]['lr'] *= 0.8
        
            self.fc.train()
            if self.args['backbone']:
                self.backbone.train()
        tb_writer.close()
        torch.save(self.backbone.state_dict(), model_out_dir + "/backbone.pt")
        torch.save(self.fc.state_dict(), model_out_dir + "/classifier.pt")
            
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

    def get_acc(self, data, name=""):
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
        print("Accuracy on {} test set: {}".format(name, top1acc))
    
        return top1acc
                
        
def get_parser():
    """
    Create parser and return arguments for experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", "-l", default=1, type=float)
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--backbone", "-bb", default=False, action='store_true')
    parser.add_argument("--lr", "-lr", default=0.1, type=float)
    parser.add_argument("--batch-size", "-b", default=64, type=int)
    parser.add_argument("--log-dir", "-d", type=str, required=True, help="Input name of dir where tb_writer"
                                                                         "output will be stored using -d flag")
    
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = vars(get_parser())
    model = EntModel(model_path="../out/toybox_baseline/", exp_args=args)
    transform_in12 = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(p=0.2),
                                         transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
    in12_test_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False, transform=transform_in12)

    in12_train_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True, transform=transform_in12,
                                                           fraction=0.2)
    
    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256),
                                           transforms.RandomResizedCrop(size=224),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)])
    
    toybox_train_data = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, rng=np.random.default_rng(0), train=True,
                                                     num_instances=20, num_images_per_class=200,
                                                     transform=transform_toybox)
    model.train_model(data=toybox_train_data, eval_data=in12_test_data, ent_train_data=in12_train_data)

    toybox_test_data = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, rng=np.random.default_rng(0), train=False,
                                                    transform=transform_toybox)
    model.get_acc(data=in12_test_data, name="in12")
    model.get_acc(data=toybox_test_data, name="toybox")
    