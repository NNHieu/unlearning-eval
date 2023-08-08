from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import mean_squared_error
import numpy as np
import torch.nn as nn

from base import AccuracyEvaluator


class ClassificationAccuracyEvaluator(AccuracyEvaluator):
    """
    Accuracy Evaluator for classification model
    """
    def __init__(self, forget_set, test_set, forget_label, test_label):
        super().__init__(forget_set, test_set, forget_label, test_label)

    def eval(self, unlearn_model: nn.Module):
        """
        Accuracy and F1 score for unlearn model in forget dataset and test dataset
        :param unlearn_model: torch neural network for classification model returning probabilities array of each class
        :return: dictionary of score
        """
        result = {}
        forget_prob = unlearn_model(self.forget_set)
        forget_predict = np.argmax(forget_prob, axis=0)
        result["forget_set"] = {
            "acc": accuracy_score(self.forget_true, forget_predict),
            "f1": f1_score(self.forget_true, forget_predict),
            "precision": precision_score(self.forget_true, forget_predict)
        }
        test_prob = unlearn_model(self.test_set)
        test_predict = np.argmax(test_prob, axis=0)
        result["test_set"] = {
            "acc": accuracy_score(self.test_true, test_predict),
            "f1": f1_score(self.test_true, test_predict),
            "precision": precision_score(self.test_set, test_predict)
        }


class RegressionAccuracyEvaluator(AccuracyEvaluator):
    """
    Accuracy Evaluator for Regression Neural Network
    """
    def __init__(self, forget_set, test_set, forget_value, test_value):
        super().__init__(forget_set, test_set, forget_value, test_value)

    def eval(self, unlearn_model: nn.Module):
        """
        L2 loss for unlearn model in forget dataset and test dataset
        :param unlearn_model: torch neural network for classification model returning estimate value
        :return: dictionary of score
        """
        forget_predict = unlearn_model(self.forget_set)
        test_predict = unlearn_model(self.test_set)
        result = {
            "forget_set": {
                "mse": mean_squared_error(self.forget_true, forget_predict)
            },
            "test_set": {
                "mse": mean_squared_error(self.test_true, test_predict)
            }
        }
        return result
