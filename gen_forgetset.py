
import torch
import torchvision
import torchvision.transforms as transforms

from utils import make_folders, random_seed




if __name__ == "__main__":
    RANDOM_SEED = 42
    random_seed(RANDOM_SEED)
    
    # Set up the CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, transform=transform, download=True)

    # Find the sample IDs with label 0
    label_to_select = 0
    sample_ids_with_label_0 = [idx for idx, (image, label) in enumerate(train_dataset) if label == label_to_select]

    # Select 5000 random sample IDs
    import random
    random.seed(42)  # Set seed for reproducibility
    selected_sample_ids = random.sample(sample_ids_with_label_0, 5000)

    # Print the selected sample IDs
    print(selected_sample_ids)
    
    # write array to .npy file
    import numpy as np
    np.save(f'forget_idx_class_{label_to_select}.npy', selected_sample_ids)