import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error
import numpy as np
import torch.nn as nn

from .base import BaseEvaluator

def accuracy(net, loader, DEVICE):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    f1 = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        f1 += f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='micro')*targets.size(0)
    return correct / total, f1 / total

class ClassificationAccuracyEvaluator(BaseEvaluator):
    """
    Accuracy Evaluator for classification model
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def eval(self, unlearn_model: nn.Module, *, device, **kwargs):
        """
        Accuracy and F1 score for unlearn model in forget dataset and test dataset
        :param unlearn_model: torch neural network for classification model returning probabilities array of each class
        :return: dictionary of score
        """
        forget_infosrc = self.infosrc['unlearned']['forget']
        test_infosrc = self.infosrc['unlearned']['test']

        result = {}
        forget_acc = forget_infosrc.accuracy
        forget_f1 = f1_score(forget_infosrc.labels.numpy(), forget_infosrc.predictions.numpy(), average='micro')*forget_infosrc.labels.size(0)
        result["forget_set"] = {
            "acc": forget_acc,
            "f1": forget_f1,
            # "len": forget_infosrc.labels.size(0)
        }

        test_acc = test_infosrc.accuracy
        test_f1 = f1_score(test_infosrc.labels.numpy(), test_infosrc.predictions.numpy(), average='micro')*test_infosrc.labels.size(0)
        result["test_set"] = {
            "acc": test_acc,
            "f1": test_f1,
            # "len": test_infosrc.labels.size(0)
        }
        return result


# class RegressionAccuracyEvaluator(AccuracyEvaluator):
#     """
#     Accuracy Evaluator for Regression Neural Network
#     """
#     def __init__(self, forget_set, test_set, forget_value, test_value):
#         super().__init__(forget_set, test_set, forget_value, test_value)

#     def eval(self, unlearn_model: nn.Module):
#         """
#         L2 loss for unlearn model in forget dataset and test dataset
#         :param unlearn_model: torch neural network for classification model returning estimate value
#         :return: dictionary of score
#         """
#         forget_predict = unlearn_model(self.forget_set)
#         test_predict = unlearn_model(self.test_set)
#         result = {
#             "forget_set": {
#                 "mse": mean_squared_error(self.forget_true, forget_predict)
#             },
#             "test_set": {
#                 "mse": mean_squared_error(self.test_true, test_predict)
#             }
#         }
#         return result
