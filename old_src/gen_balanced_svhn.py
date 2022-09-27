"""code to generate balanced svhn"""
import torchvision.datasets as datasets
import numpy as np

if __name__ == "__main__":
    train_count = 3000
    val_count = 1500
    test_count = 1500
    dataset = datasets.SVHN(root="../data", split="train", transform=None)
    test_dataset = datasets.SVHN(root="../data", split="test", transform=None)
    training_indices = []
    validation_indices = []
    test_indices = []
    train_counts = [0] * 10
    validation_counts = [0] * 10
    test_counts = [0] * 10
    rng = np.random.default_rng()
    trainset_idxs = np.arange(len(dataset))
    rng.shuffle(trainset_idxs)
    for idx in trainset_idxs:
        img, label = dataset[idx]
        if train_counts[label] < train_count:
            train_counts[label] += 1
            training_indices.append(idx)
        elif validation_counts[label] < val_count:
            validation_counts[label] += 1
            validation_indices.append(idx)
    assert len(training_indices) == len(set(training_indices))
    assert len(validation_indices) == len(set(validation_indices))
    for idx in training_indices:
        assert idx not in validation_indices
    for idx in validation_indices:
        assert idx not in training_indices
    print(train_counts, len(training_indices))
    print(validation_counts, len(validation_indices))
    testset_idxs = np.arange(len(test_dataset))
    rng.shuffle(testset_idxs)
    for idx in testset_idxs:
        img, label = test_dataset[idx]
        if test_counts[label] < test_count:
            test_counts[label] += 1
            test_indices.append(idx)
    assert len(test_indices) == len(set(test_indices))
    for idx in test_indices:
        assert idx < len(test_dataset)
    print(test_counts, len(test_indices))
    train_file_name = "../data/svhn_balanced_train_indices.npy"
    validation_file_name = "../data/svhn_balanced_val_indices.npy"
    test_file_name = "../data/svhn_balanced_test_indices.npy"
    np.save(train_file_name, np.array(training_indices))
    np.save(validation_file_name, np.array(validation_indices))
    np.save(test_file_name, np.array(test_indices))
    
    arr1, arr2, arr3 = np.load(train_file_name), np.load(validation_file_name), np.load(test_file_name)
    print(len(arr1), len(arr2), len(arr3))
    