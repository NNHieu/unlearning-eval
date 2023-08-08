from base import RetainBaseEvaluator
import torch.nn as nn
import numpy as np


class ActivationDistance(RetainBaseEvaluator):
    """
    Measure the distance between each layer of two models
    """
    def __init__(self, forget_set, test_set, base_model: nn.Module):
        super().__init__(forget_set, test_set, base_model)

    def eval(self, unlearn_model: nn.Module):
        l2_loss = 0
        norm = 0
        for (k, p), (k_base, p_base) in zip(unlearn_model.named_parameters(), self.base_model.named_parameters()):
            if p.require_grad:
                l2_loss += (p - p_base).pow(2).sum()
                norm += p.pow(2).sum()
        return {"l2_loss": l2_loss, "norm_loss": np.sqrt(l2_loss/norm)}


