import torch
from torch import nn
from unlearn_alg.finetuning import unlearning
from unlearn_alg.ssd import ssd_tuning
from functools import partial

from unlearn_eval import (
    ClassificationAccuracyEvaluator, 
    SimpleMiaEval, 
    ActivationDistance, 
    ZeroRetrainForgetting,
    Cifar10_Resnet18_Set,
    Pipeline,
    LIRA
)
from unlearn_eval.utils import InformationSource

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
print("Running on device:", DEVICE.upper())

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)

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
      ZeroRetrainForgetting(ref_model_name="dummy"),
      SimpleMiaEval(n_splits=10, random_state=0),
      # LIRA()
  ]
  # ---------------- End Init evaluators ----------------
  pipeline.set_evaluators(evaluators)

  print("Start evaluation")
  # print(pipeline.eval(unlearning))
  unlearning_fn = partial(ssd_tuning, full_train_dl=data_model_set.train_loader)
  unlearning_fn.__name__ = "ssd_tuning"
  print(pipeline.eval(unlearning_fn))
  print("Done evaluation")

if __name__ == "__main__":
  main()