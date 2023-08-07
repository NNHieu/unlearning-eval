import torch.nn as nn


class BaseEvaluator:
    forget_set = None
    test_set = None

    def __init__(self, forget_set, test_set):
        self.forget_set = forget_set
        self.test_set = test_set

    def eval(self, unlearn_model):
        raise NotImplementedError("Need to inherit")


class AccuracyEvaluator(BaseEvaluator):
    """
    Accuracy Evaluator
    """
    def __init__(self, forget_set, test_set, forget_true, test_true):
        super().__init__(forget_set, test_set)
        self.forget_true = forget_true
        self.test_true = test_true

    def eval(self, unlearn_model: nn.Module):
        raise NotImplementedError("Need to fill up")


class RetainBaseEvaluator(BaseEvaluator):
    """
    Retain-model based Evaluator
    """
    def __init__(self, forget_set, test_set, base_model: nn.Module):
        super().__init__(forget_set, test_set)
        self.base_model = base_model

    def eval(self, unlearn_model: nn.Module):
        raise NotImplementedError("Need to fill up")
