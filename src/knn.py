"""
Implement KNN classifier for trained models
"""
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as functional
import numpy as np

import dataset_imagenet12

IN12_DATA_PATH = "../data_12/IN-12/"
IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)


class KNN:
    """
    This class implements KNN classifier for given training data and test data.
    Can be subclassed for ZSL and FSL
    """
    def __init__(self, train_data, model_path, cosine=False, k=1):
        self.cosine = cosine
        self.train_data = train_data
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=64, num_workers=4, shuffle=False)
        self.k = k
        self.model_path = model_path
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        print("Loading weights from {}".format(self.model_path))
        self.backbone.load_state_dict(torch.load(self.model_path))
        self.backbone.cuda()
        self.backbone.eval()

        self.train_activations = None
        self.mean_activations_train = None
        self.train_labels = None
        self.test_data = None
        self.test_loader = None
        self.test_activations = None
        self.test_labels = None
        self.test_predictions = None
        
        self.get_train_activations()
    
    def get_train_activations(self):
        """
        This method gets the activations for the training data.
        """
        len_train_data = len(self.train_data)
        self.train_activations = torch.zeros(len_train_data, self.fc_size)
        self.train_labels = torch.zeros(len_train_data, dtype=torch.float32)
        self.mean_activations_train = torch.zeros(12, self.fc_size)
    
        for idx, images, labels in self.train_loader:
            images = images.cuda()
            with torch.no_grad():
                feats = self.backbone.forward(images)
                feats = feats.cpu()
            self.train_labels[idx] = labels.float()
            self.train_activations[idx] = feats
        for cl in range(12):
            indices_cl = torch.nonzero(self.train_labels == cl)
            activations_cl = torch.squeeze(self.train_activations[indices_cl])
            mean_act = torch.mean(activations_cl, dim=0, keepdim=False)
            self.mean_activations_train[cl] = mean_act
        
    def get_test_activations(self, test_data):
        """
        Compute the test activations from the model
        """
        self.test_data = test_data
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=64, num_workers=4, shuffle=False)
        
        self.test_activations = torch.zeros(len(self.test_data), self.fc_size)
        self.test_labels = torch.zeros(len(self.test_data), dtype=torch.float32)
        for idx, images, labels in self.test_loader:
            images = images.cuda()
            with torch.no_grad():
                feats = self.backbone.forward(images)
                feats = feats.cpu()
            self.test_labels[idx] = labels.float()
            self.test_activations[idx] = feats
                
    def get_test_predictions(self):
        """
        Calculate the test predictions from the test activations and train activations
        """
        self.test_activations = functional.normalize(self.test_activations)
        self.train_activations = functional.normalize(self.train_activations)
        self.mean_activations_train = functional.normalize(self.mean_activations_train)
        
        if self.cosine:
            distance = torch.matmul(self.test_activations, self.mean_activations_train.T)
            min_dist, min_indices = torch.topk(distance, k=1, dim=1, largest=True)
        else:
            distance = torch.cdist(self.test_activations, self.mean_activations_train)
            min_dist, min_indices = torch.topk(distance, k=1, dim=1, largest=False)
        
        # nearest_neighbors = self.train_labels[min_indices]
        self.test_predictions = min_indices  # torch.mode(nearest_neighbors, dim=1).values.numpy()
    
        correct = sum([1 for i in range(len(self.test_data)) if self.test_predictions[i] == self.test_labels[i]])
        accuracy = correct / len(self.test_data) * 100.0
        print("Accuracy is {}".format(accuracy))
        return accuracy


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                    transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
    tr_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True, transform=transform, fraction=1.0)
    backbone_path = "../out/toybox_baseline/backbone_trainer_resnet18_backbone.pt"
    knn = KNN(train_data=tr_data, model_path=backbone_path, k=19, cosine=False)
    te_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False, transform=transform, fraction=0.2)
    knn.get_test_activations(test_data=te_data)
    knn.get_test_predictions()
    