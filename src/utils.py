"""
Module for project utils
"""
import torch
import argparse
import logging


def calc_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batchSize = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target_reshaped = torch.reshape(target, (1, -1)).repeat(maxk, 1)
        correct_top_k = torch.eq(pred, target_reshaped)
        pred_1 = pred[0]
        res = []
        for k in topk:
            correct_k = correct_top_k[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(torch.mul(correct_k, 100.0 / batchSize))
        return res, pred_1
    
    
def get_parser():
    """
    Use argparse to get parser
    """
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--log-level", "-l", default="warning", type=str)
    parser.add_argument("--config-file", "-conf", default="./default_config.yaml", type=str)
    parser.add_argument("--seed", "-s", default=0, type=int)
    return parser.parse_args()


def tryAssert(fn, arg, msg):
    """
    Util method for assert followed by exception
    """
    try:
        assert fn(arg), msg
    except AssertionError:
        logging.error(msg)
        raise


def weights_init(m):
    """
    Reset weights of the network
    """
    import torch.nn as nn
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        
        
def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for _, data, _ in loader:
        # print(data.shape)
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


def get_images(images, mean, std, aug=True):
    """
    Return images from the tensor provided.
    Inverse Normalize the images if aug is True
    """
    import torchvision.transforms as transforms
    trnsfrms = []
    if aug:
        trnsfrms.append(transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]))
    import torchvision.transforms as transforms
    
    trnsfrms.append(transforms.ToPILImage())
    tr = transforms.Compose(trnsfrms)
    ims1 = []
    for img_idx in range(len(images)):
        pil_img1 = tr(images[img_idx])
        ims1.append(pil_img1)
    
    import PIL.Image as Image
    
    def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
        """hconcat images"""
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    def get_concat_h_multi_blank(im_list):
        """sdfs"""
        _im = im_list.pop(0)
        for im in im_list:
            _im = get_concat_h_blank(_im, im)
        return _im
    
    def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
        """hconcat images"""
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst
    
    def get_concat_v_multi_blank(im_list):
        """sdfs"""
        _im = im_list.pop(0)
        for im in im_list:
            _im = get_concat_v_blank(_im, im)
        return _im
    
    hor_images = []
    for i in range(len(ims1) // 8):
        if len(ims1[i * 8:min(i * 8 + 8, len(ims1))]) == 8:
            hor_images.append(get_concat_h_multi_blank(ims1[i * 8:min(i * 8 + 8, len(ims1))]))
    
    return get_concat_v_multi_blank(hor_images)

