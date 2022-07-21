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
import matplotlib.colors as colors
import numpy as np
import torch.utils.data as torchdata
import os

import dataset_imagenet12
import dataset_toybox

TOYBOX_DATA_PATH = "../data_12/Toybox/"
IN12_DATA_PATH = "../data_12/IN-12/"

IN12_MEAN = (0.485, 0.456, 0.406)
IN12_STD = (0.229, 0.224, 0.225)
TOYBOX_MEAN = (0.3499, 0.4374, 0.5199)
TOYBOX_STD = (0.1623, 1894, 1775)

color_maps = cm.get_cmap('tab20b')
COLORS = {
     0: color_maps(0),  # airplane
     1: color_maps(4),  # ball
     2: color_maps(1),  # car
     3: color_maps(8),  # cat
     4: color_maps(5),  # cup
     5: color_maps(9),  # duck
     6: color_maps(10),  # giraffe
     7: color_maps(2),  # helicopter
     8: color_maps(11),  # horse
     9: color_maps(6),  # mug
     10: color_maps(7),  # spoon
     11: color_maps(3),  # truck
     }


class UMapFromModel:
    """
    This class implements basic backbone of a model and generates activations for specified datasets
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
        self.reducer = umap.UMAP(
            n_neighbors=10,
            min_dist=0.1,
            n_components=2,
            metric='cosine'
        )
        
    def reset_reducer(self, n_nbrs, min_d, metric):
        """
        This method resets the reducer based on the provided arguments
        """
        self.reducer = umap.UMAP(n_neighbors=n_nbrs, min_dist=min_d, metric=metric, n_components=2)
        
    def get_activations(self, data):
        """
        This method gets the activations for the training data.
        """
        len_train_data = len(data)
        data_loader = torchdata.DataLoader(data, batch_size=256, shuffle=False, num_workers=4)
        
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
        # all_activations = torch.cat([activations, mean_activations], dim=0)
        return activations.numpy(), labels.numpy()
    
    def fit_data(self, activations, labels=None):
        """
        Fit the UMAP reducer to the activation and return embedding
        """
        self.reducer.fit(activations, labels)
        embedding = self.reducer.transform(activations)
        return embedding
    
    def transform_data(self, activations):
        """
        Get embeddings for activations using reducer fit previously
        """
        embedding = self.reducer.transform(activations)
        return embedding
    
    @staticmethod
    def plot(embeddings, labels, markers, out_path):
        """
        Draw the scatter plot for the embeddings
        """
        classes_ordered_by_color = ['airplane', 'car', 'helicopter', 'truck', 'ball', 'cup', 'mug', 'spoon', 'cat',
                                    'duck', 'giraffe', 'horse']
        cmap = colors.ListedColormap([color_maps(i) for i in range(12)])
        bounds = list(COLORS.keys()) + [12]
        norm = colors.BoundaryNorm(bounds, 12)

        fig, ax = plt.subplots(dpi=600)
        for i in range(len(embeddings)):
            ax.scatter(embeddings[i][:, 0], embeddings[i][:, 1], marker=markers[i], s=2,
                       c=[COLORS[int(labels[i][idx])] for idx in range(len(labels[i]))])

        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical')
        cb.set_ticks(ticks=np.arange(12) + 0.5, labels=classes_ordered_by_color)
        fig.set_tight_layout(True)
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()


def get_umap_two_datasets(model_path, out_dir_path, data_1, data_2):
    """
    This method can be used to get the UMAPS for two datasets.
    The UMAP transform will be learned based on data_1 and data_2 will be treated
    as test data.
    """
    nbrs = [5, 10, 20, 50, 75, 100, 200]
    min_ds = [0.02, 0.05, 0.1, 0.2, 0.5]
    os.makedirs(out_dir_path, exist_ok=True)
    
    mapper = UMapFromModel(model_path=model_path)
    data_1_activations, data_1_labels = mapper.get_activations(data=data_1)
    data_2_activations, data_2_labels = mapper.get_activations(data=data_2)
    for nbr in nbrs:
        for d in min_ds:
            mapper.reset_reducer(n_nbrs=nbr, min_d=d, metric='cosine')
            data_1_embeddings = mapper.fit_data(activations=data_1_activations, labels=data_1_labels)
            data_2_embeddings = mapper.transform_data(activations=data_2_activations)
            plot_path = out_dir_path + "_".join(["umap", str(nbr), str(d), "cosine"]) + ".png"
            mapper.plot(embeddings=[data_1_embeddings, data_2_embeddings], labels=[data_1_labels, data_2_labels],
                        markers=['.', '+'], out_path=plot_path)
            print("Saved plot into {}".format(plot_path))
    

if __name__ == "__main__":
    transform_in12 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                         transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
    in12_train_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True, transform=transform_in12,
                                                           fraction=1.0)
    in12_test_data = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False, transform=transform_in12)
    
    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                           transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)])

    toybox_train_data = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, rng=np.random.default_rng(0), train=True,
                                                     num_instances=20, num_images_per_class=2000,
                                                     transform=transform_toybox)
    toybox_test_data = dataset_toybox.ToyboxDataset(root=TOYBOX_DATA_PATH, rng=np.random.default_rng(0), train=False,
                                                    transform=transform_toybox)
    get_umap_two_datasets(model_path="../out/in12_baseline/backbone_trainer_resnet18_backbone.pt",
                          out_dir_path="../out/in12_baseline/umap/",
                          data_1=in12_train_data, data_2=in12_test_data)
    '''
    model = UMapFromModel(model_path="../out/simclr_trial_2/ssl_resnet18_backbone.pt")
    in12_train_activations, in12_train_labels = model.get_activations(data=in12_train_data)
    in12_test_activations, in12_test_labels = model.get_activations(data=in12_test_data)
    # toybox_train_activations, toybox_train_labels = model.get_activations(data=toybox_train_data)
    # print(toybox_train_activations.shape)
    # toybox_test_activations, toybox_test_labels = model.get_activations(data=toybox_test_data)
    # print(toybox_test_activations.shape)
    train_embeddings = model.fit_data(in12_train_activations)
    test_embeddings = model.transform_data(in12_test_activations)

    model.plot(embeddings=[train_embeddings, test_embeddings], labels=[in12_train_labels, in12_test_labels],
               markers=['.', 'x'])
    '''