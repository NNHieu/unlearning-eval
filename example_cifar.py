import torch
from torch import nn
from torch import optim
from torchvision.models import resnet18
from utils import make_folders, random_seed
import logging

from unlearn_eval import (
    ClassificationAccuracyEvaluator, 
    SimpleMiaEval, 
    ActivationDistance, 
    ZeroRetrainForgetting,
    LIRA,
    Cifar10_Resnet18_Set,
    Pipeline
)
logger = logging.getLogger('logger')

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RANDOM_SEED = 42
RNG = torch.Generator().manual_seed(RANDOM_SEED)

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

def prob_unlearn(net, retain, forget, validation):
  
  
    forget_epochs = 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=forget_epochs)
    net.train()
    for id_ep in range(forget_epochs):
        running_loss = 0
        for inputs, targets in forget:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            
            log_softmax_outputs = nn.LogSoftmax(dim=1)(outputs)
            
            # Create labels with equal values of 1 / num_classes
            batch_size = inputs.size(0)
            num_classes = log_softmax_outputs.size(1)
            equal_labels = torch.full((batch_size, num_classes), 1 / num_classes, device=DEVICE)
            # import IPython; IPython.embed()
            # Convert equal value labels to log probabilities
            log_prob_labels = torch.log(equal_labels)
            # import IPython; IPython.embed()
            # exit(0)
            
            loss = nn.KLDivLoss(reduction='batchmean', log_target=True)(log_softmax_outputs, log_prob_labels)  # Calculate KLDivLoss
            if torch.any(torch.isnan(loss)):
                print("labels: ")
                print((torch.abs(log_prob_labels) < 1e-9).nonzero(as_tuple=True))
                raise Exception("start error")
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        scheduler.step()
        # print(f"Epoch: {id_ep} Loss: {running_loss}")  
    # return net
    retain_epochs = 5
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=retain_epochs)

    for _ in range(retain_epochs):
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

def rejoin_unlearn(net, retain, forget, validation):
    
    rejoin_epoch = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rejoin_epoch)
    net.train()

    for _ in range(rejoin_epoch):
        # for inputs, targets in retain:
        #     inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        #     optimizer.zero_grad()
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets)
        #     loss.backward()
        #     optimizer.step()
        # scheduler.step()

        for inputs, targets in forget:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return net

def main(unlearn_id_path="/home/hpc/phinv/unlearning-eval/data/cifar10_forget_idx_class_0_4000__1_1000.npy"):
  logger.warning(f"Start unlearn in file: {unlearn_id_path}")

  data_model_set = Cifar10_Resnet18_Set(data_root='./data/cifar10', 
                                        data_plit_RNG=RNG,
                                        index_local_path=unlearn_id_path,
                                        # model_path='./models/weights_resnet18_cifar10.pth',
                                        model_path='./models/weights_resnet18_cifar10.pth',
                                        # model_path='./models/retrain_weights_resnet18_cifar10.pth',
                                        # /home/hpc/phinv/unlearning-eval/models/retrain_weights_resnet18_cifar10.pth
                                        download_index=True)
  pipeline = Pipeline(DEVICE, RNG, data_model_set)
#   pipeline.training_retain_from_scratch(retain_epochs=100)
#   exit(0)


  # ---------------- Begin Init evaluators ----------------
  forget_loader, retain_loader, test_loader, val_loader = data_model_set.get_dataloader(RNG)
  
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
      # ActivationDistance(forget_loader, test_loader, retrained_model),
      ZeroRetrainForgetting(forget_images, test_images, retrained_model).set_norm(True),
      # ZeroRetrainForgetting(forget_images, test_images, dummy_model).set_norm(True),
      LIRA(forget_loader, test_loader, data_model_set.forget_idx, data_model_set.train_set, data_model_set.train_set, num_exps=16),
      # SimpleMiaEval(forget_loader, test_loader, nn.CrossEntropyLoss(reduction="none"), n_splits=10, random_state=0)
  ]
  # ---------------- End Init evaluators ----------------
  pipeline.set_evaluators(evaluators)

#   print("Retrained model accuracy: ")
  # retrain_eval = ClassificationAccuracyEvaluator(forget_loader, test_loader, None, None)
  # res = retrain_eval.eval(dummy_model, device=DEVICE)
  # print(res)

  logger.warning("Start evaluation")
  # print(pipeline.eval(unlearning))
  print(pipeline.eval(prob_unlearn))
  # ft_model = pipeline.eval(prob_unlearn)
  print("Done prob unlearn")

  # data_model_set.set_model(ft_model)
  # rejoin_model = pipeline.eval(rejoin_unlearn)
  
  # print("Done rejoin unlearn")

if __name__ == "__main__":
  random_seed(RANDOM_SEED)
  make_folders()
  import glob
  import os
  list_unlearn_id_path = glob.glob(os.path.join("/home/hpc/phinv/unlearning-eval/data", "**", "*.npy"), recursive=True)
  list_unlearn_id_path = sorted(list_unlearn_id_path)
  # print(list_unlearn_id_path)
  for id_file, unlearn_id_path in enumerate(list_unlearn_id_path):
    # print(id_file, unlearn_id_path)
    # if id_file in [1, 5, 10]:
    main(unlearn_id_path)
    #
  exit(0)
  main()
  exit(0)

  print("Retrained model accuracy:")
  retrain_eval = ClassificationAccuracyEvaluator(forget_loader, test_loader, None, None)
  res = retrain_eval.eval(retrained_model, device=DEVICE)
  print(res)

  print("Start evaluation")
  print(pipeline.eval(unlearning))
  print("Done evaluation")
