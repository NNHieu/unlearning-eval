import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18


import AccuracyEvaluator
from RetainBasedEvaluator import ActivationDistance, ZeroRetrainForgetting
from attack_based import SimpleMiaEval, simple_mia

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)

def unlearning(net, retain, forget, validation):
    """Unlearning by fine-tuning.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """
    epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    return net

def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total

def prepare_data():
    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize
    )
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

    # download the forget and retain index split
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # split train set into a forget and a retain set
    forget_set = torch.utils.data.Subset(train_set, forget_idx)
    retain_set = torch.utils.data.Subset(train_set, retain_idx)

    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=128, shuffle=True, num_workers=2
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )
    return forget_loader, retain_loader, test_loader, val_loader

def get_pretrained_model():
    # download pre-trained weights
    local_path = "weights_resnet18_cifar10.pth"
    # if not os.path.exists(local_path):
    #     response = requests.get(
    #         "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
    #     )
    #     open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    return model

def main(unlearning_fn, model, dataloaders, evaluators):
    forget_loader, retain_loader, test_loader, val_loader = dataloaders

    ft_model = copy.deepcopy(model)

    # Execute the unlearing routine. This might take a few minutes.
    # If run on colab, be sure to be running it on  an instance with GPUs
    ft_model = unlearning_fn(ft_model, retain_loader, forget_loader, test_loader)

    print(f"Retain set accuracy: {100.0 * accuracy(ft_model, retain_loader):0.1f}%")
    print(f"Test set accuracy: {100.0 * accuracy(ft_model, test_loader):0.1f}%")

    # evaluate the unlearned model
    results = {}
    # zeroretain_evaluator = ZeroRetrainForgetting(forget_set, test_set, model)
    # retain_evaluator = ActivationDistance(None, None, model)
    # evaluator = AccuracyEvaluator.ClassificationAccuracyEvaluator(None, None, None, None)
    for evaluator in evaluators:
        results[evaluator.__class__.__name__] = evaluator.eval(ft_model, device=DEVICE)
    
    return results

if __name__ == "__main__":
    dataloaders = prepare_data()
    forget_loader, retain_loader, test_loader, val_loader = dataloaders
    
    forget_images = []
    for i in forget_loader.dataset:
        forget_images.append(i[0])
    forget_images = torch.stack(forget_images)

    test_images = []
    for i in test_loader.dataset:
        test_images.append(i[0])
    test_images = torch.stack(test_images)

    model = get_pretrained_model()
    model.eval()
    
    evaluators = [
        AccuracyEvaluator.ClassificationAccuracyEvaluator(forget_loader, test_loader, None, None),
        ActivationDistance(forget_loader, test_loader, model),
        ZeroRetrainForgetting(forget_images, test_images, model),
        SimpleMiaEval(forget_loader, test_loader, nn.CrossEntropyLoss(reduction="none"), n_splits=10, random_state=0)
    ]

    results = main(unlearning, model, dataloaders, evaluators)
    print(results)