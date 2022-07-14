"""
Module to get UMAP visualization
"""

import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dataset_imagenet12

IN12_DATA_PATH = "../data_12/IN-12/"
IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)

toybox_classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon',
                  'truck']


class Model:
    """
    This class implements KNN classifier for given training data and test data.
    Can be subclassed for ZSL and FSL
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.backbone = models.resnet18(pretrained=False)
        self.fc_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        print("Loading weights from {}".format(self.model_path))
        self.backbone.load_state_dict(torch.load(self.model_path))
        self.backbone.cuda()
        self.backbone.eval()
        
    def get_activations(self, data):
        """
        This method gets the activations for the training data.
        """
        len_train_data = len(data)
        data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False, num_workers=4)
        
        activations = torch.zeros(len_train_data, self.fc_size)
        labels = torch.zeros(len_train_data, dtype=torch.float32)
        mean_activations = torch.zeros(12, self.fc_size)
        
        for idx, images, labls in data_loader:
            images = images.cuda()
            with torch.no_grad():
                feats = self.backbone.forward(images)
                feats = feats.cpu()
            labels[idx] = labls.float()
            activations[idx] = feats
        for cl in range(12):
            indices_cl = torch.nonzero(labels == cl)
            activations_cl = torch.squeeze(activations[indices_cl])
            mean_act = torch.mean(activations_cl, dim=0, keepdim=False)
            mean_activations[cl] = mean_act
        all_activations = torch.cat([activations, mean_activations], dim=0)
        print(activations.shape, mean_activations.shape, all_activations.shape)
        return activations.numpy(), labels.numpy()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                    transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
    in12_train_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True, transform=transform,
                                                           fraction=0.05)
    color_maps = cm.get_cmap('tab20b')
    COLORS = {
        0: color_maps(0),   # airplane
        1: color_maps(4),   # ball
        2: color_maps(1),   # car
        3: color_maps(8),   # cat
        4: color_maps(5),   # cup
        5: color_maps(9),   # duck
        6: color_maps(10),  # giraffe
        7: color_maps(2),    # helicopter
        8: color_maps(11),    # horse
        9: color_maps(6),    # mug
        10: color_maps(7),   # spoon
        11: color_maps(3),   # truck
    }
    
    model = Model(model_path="../out/in12_baseline/backbone_trainer_resnet18_backbone.pt")
    in12_train_activations, in12_train_labels = model.get_activations(data=in12_train_data)
    reducer = umap.UMAP()
    reducer.fit(in12_train_activations)
    embedding = reducer.transform(in12_train_activations)
    plt.scatter(embedding[:, 0], embedding[:, 1], marker='.',
                c=[COLORS[int(in12_train_labels[i])] for i in range(len(in12_train_labels))])
    import numpy as np
    # plt.colorbar(boundaries=np.arange(13)-0.5)
    plt.show()
            
    
