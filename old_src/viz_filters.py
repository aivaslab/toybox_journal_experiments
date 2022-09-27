"""
Code to visualize conv1 filters from resnet-18
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn


def generate_filters(model_name):
    """Generate image with filters and save it"""
    out_img_name = model_name + ".png"
    net = models.resnet18(pretrained=False)
    net.fc = nn.Identity()
    net.load_state_dict(torch.load(model_name))
    model_children = list(net.children())
    print(model_children[0], type(model_children[0]), model_children[0].weight.shape)
    
    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))
    for i, fltr in enumerate(model_children[0].weight):
        plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        fltr = fltr.detach().numpy()
        fltr = fltr - fltr.min()
        fltr = fltr / fltr.max()
        fltr = np.transpose(fltr, (1, 2, 0))
        plt.imshow(fltr)
        plt.axis('off')
        plt.savefig(out_img_name)
    plt.show()


generate_filters("../out/in12_baseline/backbone_trainer_resnet18_backbone.pt")
