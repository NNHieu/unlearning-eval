import os
import requests
import numpy as np
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import tqdm
import logging
logger = logging.getLogger('logger')

from unlearn_eval import (
    ClassificationAccuracyEvaluator, 
    SimpleMiaEval, 
    ActivationDistance, 
    ZeroRetrainForgetting
)

def unlearning(net, retain, forget, validation):
    return net

def accuracy(net, loader, DEVICE):
    """Return accuracy on a dataset given by the data loader."""
    total_loss, total_sample, total_acc = 0, 0, 0
    net.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total_loss = nn.CrossEntropyLoss()(outputs, targets)
        total_sample += targets.size(0)
        total_acc += predicted.eq(targets).sum().item()
    return (total_loss/ total_sample, total_acc / total_sample, total_sample)
    # return correct / total

class Cifar10_Resnet18_Set():

    def __init__(self, data_root, data_plit_RNG, index_local_path, model_path, download_index=False) -> None:
        self._prepare_main_datasets(data_root=data_root, RNG=data_plit_RNG)
        
        self._generate_forget_set(index_local_path=index_local_path, download_index=download_index)
        
        print("Done generate forget set")
        
        self.forget_loader, self.retain_loader, self.test_loader, self.val_loader = self.get_dataloader(data_plit_RNG)
        
        self._prepare_pretrained_weights(local_path=model_path, download=False)
        print("Download pretrained weights")

    def _prepare_main_datasets(self, data_root="./data", RNG=42):
         # download and pre-process CIFAR10
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.train_set = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=normalize
        )
        # we split held out data into test and validation set
        held_out = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=normalize
        )
        self.test_set, self.val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
        
        # self._generate_forget_set()
        
        
        
    def _generate_forget_set(self, download_index=False, index_local_path="forget_idx.npy"):
        # download the forget and retain index split
        if download_index:
            if not os.path.exists(index_local_path):
                response = requests.get(
                    "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + "forget_idx.npy"
                )
                open(index_local_path, "wb").write(response.content)
            else:
                print("Index file already downloaded")
        else:
            if not os.path.exists(index_local_path): 
                raise RuntimeError("Index file is not exist")
        forget_idx = np.load(index_local_path)

        # construct indices of retain from those of the forget set
        forget_mask = np.zeros(len(self.train_set.targets), dtype=bool)
        forget_mask[forget_idx] = True
        retain_idx = np.arange(forget_mask.size)[~forget_mask]

        # split train set into a forget and a retain set
        self.forget_set = torch.utils.data.Subset(self.train_set, forget_idx)
        self.retain_set = torch.utils.data.Subset(self.train_set, retain_idx)
        
        label_forget_set = [label for idx, (_, label) in enumerate(self.forget_set) ]
        logger.warning(f"Label distribution of forget set: {np.unique(label_forget_set, return_counts=True)}")


    def get_dataloader(self, RNG):
        train_loader = DataLoader(self.train_set, batch_size=128, shuffle=True, num_workers=2)
        test_loader = DataLoader(self.test_set, batch_size=128, shuffle=False, num_workers=2)
        val_loader = DataLoader(self.val_set, batch_size=128, shuffle=False, num_workers=2)
        forget_loader = DataLoader(
            self.forget_set, batch_size=128, shuffle=True, num_workers=2
        )
        retain_loader = DataLoader(
            self.retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
        )
        return forget_loader, retain_loader, test_loader, val_loader
    
    def _download_pretrained_model(self, save_path):
        response = requests.get(
                    "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
                )
        open(save_path, "wb").write(response.content)

    def _prepare_pretrained_weights(self,local_path = "weights_resnet18_cifar10.pth",  download=False, ):
        # download pre-trained weights
        if download:
            if not os.path.exists(local_path):
                self._download_pretrained_model(save_path=local_path)
            logger.info(f"Download pre-trained weights from local file: {local_path}")
        logger.info(f"Set weights_pretrained from local file: {local_path}")
        self.weights_pretrained = torch.load(local_path)

    def get_init_model(self):
        logger.info(f"Init model without pretrained weights")
        model = resnet18(weights=None, num_classes=10)
        # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # in_features = model.fc.in_features
        # model.fc = nn.Linear(in_features, 10)
        # model = resnet18(pretrained=True, num_classes=10)
        return model

    def get_pretrained_model(self):
        # load model with pre-trained weights
        # TODO, check acc w-o pretrained model
        # return self.get_init_model()
        model = resnet18(weights=None, num_classes=10)
        model.load_state_dict(self.weights_pretrained)
        return model
    
    def get_retrained_model(self):
        '''
        TODO: download a real retrained model.
        In this version, this function just return the original model.
        '''
        return self.get_pretrained_model()


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