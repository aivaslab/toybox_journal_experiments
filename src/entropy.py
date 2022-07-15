"""
Module that implements method that learns from entropy on target distribution
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np

import dataset_imagenet12
import dataset_toybox

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
    def __init__(self, model_path):
        self.model_path = model_path
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(self.fc_size, 12)
        self.softmax = nn.Softmax(dim=1)
        
        backbone_path = self.model_path + "backbone_trainer_resnet18_backbone.pt"
        classifier_path = self.model_path + "backbone_trainer_resnet18_classifier.pt"
        
        self.backbone.load_state_dict(torch.load(backbone_path))
        self.fc.load_state_dict(torch.load(classifier_path))
        
        self.backbone.eval()
        self.backbone.cuda()
        self.fc.eval()
        self.fc.cuda()
        
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
        data_loader = torchdata.DataLoader(data, batch_size=128, num_workers=4, pin_memory=True,
                                           persistent_workers=True)
        total_entropy = 0.0
        batches = 0
        for _, images, _ in data_loader:
            images = images.cuda()
            with torch.no_grad():
                feats = self.backbone(images)
                activations = self.fc(feats)
                batch_entropy = self.calc_entropy(activations)
                total_entropy += torch.mean(batch_entropy)
                batches += 1
        print("Average Entropy: {}".format(total_entropy/batches))
                
        
if __name__ == "__main__":
    model = EntModel(model_path="../out/toybox_baseline/")
    transform_in12 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                         transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
    in12_test_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False, transform=transform_in12)
    model.run_dataset(data=in12_test_data)
    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                           transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)])
    toybox_train_data = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, rng=np.random.default_rng(), train=True,
                                                     num_instances=10, num_images_per_class=1000,
                                                     transform=transform_toybox)
    model.run_dataset(data=toybox_train_data)
    