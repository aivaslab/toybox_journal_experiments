"""
Dataset class for MNIST-50
"""
import torchvision.transforms as transforms
import torch.utils.data
import idx2numpy
import numpy as np
import cv2

OUT_PATH = "../data/"


def split_mnist(path="../data/", train_images_per_class=50, val_images_per_class=1000):
    """split mnist training set into mnist-50 and validation"""
    training_images_file = path + "MNIST/raw/train-images-idx3-ubyte"
    training_labels_file = path + "MNIST/raw/train-labels-idx1-ubyte"
    images = idx2numpy.convert_from_file(training_images_file)
    labels = idx2numpy.convert_from_file(training_labels_file)
    rng = np.random.default_rng(seed=108)
    all_indices = np.arange(labels.shape[0])
    rng.shuffle(all_indices)
    train_counts = [0] * 10
    val_counts = [0] * 10
    train_found = 0
    val_found = 0
    idx = 0
    train_indices = []
    val_indices = []
    while train_found < train_images_per_class * 10 or val_found < val_images_per_class * 10:
        label = labels[idx]
        if train_counts[label] < train_images_per_class:
            train_counts[label] += 1
            train_found += 1
            train_indices.append(idx)
        elif val_counts[label] < val_images_per_class:
            val_counts[label] += 1
            val_found += 1
            val_indices.append(idx)
        idx += 1
    all_labels = []
    for idx in train_indices:
        lbl = labels[idx]
        all_labels.append(lbl)
    import collections
    cntr = collections.Counter(all_labels)
    print(cntr)
    all_labels = []
    for idx in val_indices:
        lbl = labels[idx]
        all_labels.append(lbl)
    import collections
    cntr = collections.Counter(all_labels)
    print(cntr)
    for idx in val_indices:
        assert idx not in train_indices
    print(images.dtype)
    train_images = np.zeros((len(train_indices), 28, 28), dtype=np.uint8)
    train_labels = np.ones(len(train_indices)) * -1
    for i, img_idx in enumerate(train_indices):
        img = images[img_idx]
        lbl = labels[img_idx]
        train_images[i] = img
        train_labels[i] = lbl
    train_images_out_file = OUT_PATH + "mnist50_train_images.npy"
    train_labels_out_file = OUT_PATH + "mnist50_train_labels.npy"
    np.save(train_images_out_file, train_images)
    np.save(train_labels_out_file, train_labels)

    val_images = np.zeros((len(val_indices), 28, 28), dtype=np.uint8)
    val_labels = np.ones(len(val_indices)) * -1
    for i, img_idx in enumerate(val_indices):
        img = images[img_idx]
        lbl = labels[img_idx]
        val_images[i] = img
        val_labels[i] = lbl
    val_images_out_file = OUT_PATH + "mnist50_val_images.npy"
    val_labels_out_file = OUT_PATH + "mnist50_val_labels.npy"
    np.save(val_images_out_file, val_images)
    np.save(val_labels_out_file, val_labels)
    
    
def display_ims(digit, train=True):
    """
    Display images for digit
    """
    if train:
        images_file = OUT_PATH + "mnist50_train_images.npy"
        labels_file = OUT_PATH + "mnist50_train_labels.npy"
    else:
        images_file = OUT_PATH + "mnist50_val_images.npy"
        labels_file = OUT_PATH + "mnist50_val_labels.npy"
    images = np.load(images_file)
    labels = np.load(labels_file)
    found = 0
    idx = 0
    indices = []
    while found < 50:
        if labels[idx] == digit:
            indices.append(idx)
            found += 1
        idx += 1
    print(found)
    all_images = np.zeros((140, 280), dtype=np.uint8)
    for i in range(50):
        row = i // 10
        col = i % 10
        img = images[indices[i]]
        all_images[row*28:(row+1)*28, col*28:(col+1)*28] = img
    cv2.imshow(str(digit), all_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # split_mnist(train_images_per_class=50)
    for i in range(10):
        display_ims(i, train=False)
    