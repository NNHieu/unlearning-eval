import torch
from torch import nn
from torch import optim
from torchvision.models import resnet18


from unlearn_eval import (
    ClassificationAccuracyEvaluator, 
    SimpleMiaEval, 
    ActivationDistance, 
    ZeroRetrainForgetting,
    Cifar10_Resnet18_Set,
    Pipeline
)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
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

def main():
  data_model_set = Cifar10_Resnet18_Set(data_root='data/cifar10', 
                                        data_plit_RNG=RNG,
                                        index_local_path='data/cifar10/forget_idx.npy',
                                        model_path='data/cifar10/weights_resnet18_cifar10.pth',
                                        download_index=False)
  pipeline = Pipeline(DEVICE, RNG, data_model_set)

  # ---------------- Begin Init evaluators ----------------
  forget_loader, _, test_loader, _ = data_model_set.get_dataloader(RNG)
  retrained_model = data_model_set.get_retrained_model()
  retrained_model.to(DEVICE)

  forget_images = []
  for i in forget_loader.dataset:
      forget_images.append(i[0])
  forget_images = torch.stack(forget_images)
  forget_images = forget_images.to(DEVICE)

  test_images = []
  for i in test_loader.dataset:
      test_images.append(i[0])
  test_images = torch.stack(test_images)
  test_images = forget_images.to(DEVICE)

  dummy_model = resnet18(num_classes=10).to(DEVICE)
  
  evaluators = [
      ClassificationAccuracyEvaluator(forget_loader, test_loader, None, None),
      ActivationDistance(forget_loader, test_loader, retrained_model),
      ZeroRetrainForgetting(forget_images, test_images, retrained_model).set_norm(True),
      SimpleMiaEval(forget_loader, test_loader, nn.CrossEntropyLoss(reduction="none"), n_splits=10, random_state=0)
  ]
  # ---------------- End Init evaluators ----------------
  pipeline.set_evaluators(evaluators)

  print("Retrained model accuracy:")
  retrain_eval = ClassificationAccuracyEvaluator(forget_loader, test_loader, None, None)
  res = retrain_eval.eval(retrained_model, device=DEVICE)
  print(res)

  print("Start evaluation")
  print(pipeline.eval(unlearning))
  print("Done evaluation")

if __name__ == "__main__":
  main()
