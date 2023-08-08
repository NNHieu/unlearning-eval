from base import RetainBaseEvaluator
import torch
import torch.nn as nn
import numpy as np
from scipy.special import rel_entr


class ActivationDistance(RetainBaseEvaluator):
    """
    Measure the distance between each layer of two models
    """
    def __init__(self, forget_set, test_set, base_model: nn.Module):
        super().__init__(forget_set, test_set, base_model)

    def eval(self, unlearn_model: nn.Module, **kwargs):
        l2_loss = 0
        norm = 0
        for (k, p), (k_base, p_base) in zip(unlearn_model.named_parameters(), self.base_model.named_parameters()):
            # print(k)
            if p.requires_grad:
                l2_loss += (p - p_base).pow(2).sum().cpu()
                norm += p.pow(2).sum().cpu()
        return {"l2_loss": l2_loss, "norm_loss": torch.sqrt(l2_loss/norm)}


class ZeroRetrainForgetting(RetainBaseEvaluator):
    """
    Measure the randomness in the model's prediction by comparing them with based model 
    In other words: Measure the change in accuracy on the unlearned data after retraining the model without the data to be unlearned 
    Used for classification neural network only
    """
    def __init__(self, forget_set, test_set, base_model: nn.Module):
        super().__init__(forget_set, test_set, base_model)

    @torch.no_grad()
    def eval(self, unlearn_model: nn.Module, *, device, **kwargs):
        self.forget_set = self.forget_set.to(device)

        unlearn_forget_prob = unlearn_model(self.forget_set)
        unlearn_forget_prob = torch.softmax(unlearn_forget_prob, dim=1).cpu().numpy()
        
        base_forget_prob = self.base_model(self.forget_set)
        base_forget_prob = torch.softmax(base_forget_prob, dim=1).cpu().numpy()


        zrf = 0
        for idx in range(len(self.forget_set)):
            avg = 0.5 * (unlearn_forget_prob[idx] + base_forget_prob[idx])
            js = 0.5 * (rel_entr(unlearn_forget_prob[idx], avg) + rel_entr(base_forget_prob[idx], avg))
            zrf += js
        print(zrf)
        zrf = zrf / len(self.forget_set)
        return 1 - zrf