import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from unlearn_eval import (
    ClassificationAccuracyEvaluator, 
    SimpleMiaEval, 
    ActivationDistance, 
    ZeroRetrainForgetting,
    Cifar10_Resnet18_Set,
    Pipeline
)
from unlearn_eval.utils import InformationSource

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
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

    for _ in tqdm(range(epochs), desc="Unlearning"):
        for inputs, targets in tqdm(retain, desc="Finetuning", leave=False):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    return net

def main():
  data_model_set = Cifar10_Resnet18_Set(data_root='data/cifar10', 
                                        data_plit_RNG=RNG,
                                        index_local_path='data/cifar10/forget_idx.npy',
                                        model_path='data/cifar10/weights_resnet18_cifar10.pth',
                                        retrain_model_path='data/cifar10/retrain_weights_resnet18_cifar10.pth',
                                        download_index=False,
                                        criterion=nn.CrossEntropyLoss(reduction='none'))
  pipeline = Pipeline(DEVICE, RNG, data_model_set)
  retrained_model = data_model_set.get_retrained_model()
  retrained_model.to(DEVICE)

  evaluators = [
      ClassificationAccuracyEvaluator(),
      ActivationDistance(retrained_model),
      ZeroRetrainForgetting(),
      SimpleMiaEval(n_splits=10, random_state=0)
  ]
  # ---------------- End Init evaluators ----------------
  pipeline.set_evaluators(evaluators)

  print("Start evaluation")
  print(pipeline.eval(unlearning))
  print("Done evaluation")

if __name__ == "__main__":
  main()