
import torch
import torchvision
import torchvision.transforms as transforms

import random
import numpy as np

if __name__ == "__main__":
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Set up the CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, transform=None, download=True)

    # Find the sample IDs with label 0
    # label_to_select = 0
    # sample_ids_with_label_0 = [idx for idx, (image, label) in enumerate(train_dataset) if label == label_to_select]

    # Select 5000 random sample IDs
    random.seed(42)  # Set seed for reproducibility
    # selected_sample_ids = random.sample(sample_ids_with_label_0, 5000)

    # Print the selected sample IDs
    # print(selected_sample_ids)
    # write array to .npy file
    # np.save(f'forget_idx_class_{label_to_select}.npy', selected_sample_ids)

    # generate array list contains number of samples for each class. The sum of all elements in the array is 5000
    # samples_forget_set = [500] * 10

    labels_list = np.array([label for _, label in train_dataset])

    # samples_forget_set = [5000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # samples_forget_set = [4000, 1000, 0, 0, 0, 0, 0, 0, 0, 0]
    # samples_forget_set = [3000, 1000, 1000, 0, 0, 0, 0, 0, 0, 0]
    # samples_forget_set = [2000, 1000, 1000, 1000, 0, 0, 0, 0, 0, 0]
    # samples_forget_set = [1000, 1000, 1000, 1000, 1000, 0, 0, 0, 0, 0]
    # samples_forget_set = [1000, 1000, 1000, 1000, 500, 500, 0, 0, 0, 0]
    # samples_forget_set = [1000, 1000, 1000, 500, 500, 500, 500, 0, 0, 0]
    # samples_forget_set = [1000, 1000, 500, 500, 500, 500, 500, 500, 0, 0]
    # samples_forget_set = [1000, 500, 500, 500, 500, 500, 500, 500, 500, 0]
    # samples_forget_set = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

    samples_forget_set = []
    samples_forget_set.append([5000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    samples_forget_set.append([4000, 1000, 0, 0, 0, 0, 0, 0, 0, 0])
    samples_forget_set.append([3000, 1000, 1000, 0, 0, 0, 0, 0, 0, 0])
    samples_forget_set.append([2000, 1000, 1000, 1000, 0, 0, 0, 0, 0, 0])
    samples_forget_set.append([1000, 1000, 1000, 1000, 1000, 0, 0, 0, 0, 0])
    samples_forget_set.append([1000, 1000, 1000, 1000, 500, 500, 0, 0, 0, 0])
    samples_forget_set.append([1000, 1000, 1000, 500, 500, 500, 500, 0, 0, 0])
    samples_forget_set.append([1000, 1000, 500, 500, 500, 500, 500, 500, 0, 0])
    samples_forget_set.append([1000, 500, 500, 500, 500, 500, 500, 500, 500, 0])
    samples_forget_set.append([500, 500, 500, 500, 500, 500, 500, 500, 500, 500])

    for i in range(len(samples_forget_set)):
        samples_forget_set_i = np.array(samples_forget_set[i])
        print(np.sum(samples_forget_set_i), samples_forget_set_i)
    

        index_forget_set = []

        for label in range(10):
            if samples_forget_set_i[label] == 0:
                continue
            index_labels = np.where(labels_list == label)[0]
            # print(len(index_labels), index_labels[:10], train_dataset[index_labels[0]][1])
            selected_sample_ids = random.sample(list(index_labels), samples_forget_set_i[label])
            index_forget_set.extend(selected_sample_ids)

        index_forget_set = sorted(index_forget_set)
        print(len(index_forget_set), index_forget_set[:10])
        np.save(f'./data/cifar10_forget_idx_{"_".join([str(x) for x in samples_forget_set_i])}.npy', index_forget_set)

    # np.save(f'./data/cifar10_forget_idx_class_0_5000.npy', index_forget_set)
    # np.save(f'./data/cifar10_forget_idx_class_0_4000__1_1000.npy', index_forget_set)