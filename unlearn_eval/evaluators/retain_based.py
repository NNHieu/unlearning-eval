from .base import BaseEvaluator, RetainBaseEvaluator
import torch
import torch.nn as nn
import numpy as np
from scipy.special import rel_entr
from torch.nn.functional import kl_div


class ActivationDistance(RetainBaseEvaluator):
    """
    Measure the distance between each layer of two models
    """
    def __init__(self, base_model: nn.Module):
        super().__init__(base_model)

    @torch.no_grad()
    def eval(self, unlearn_model: nn.Module, **kwargs):
        l2_loss = 0
        norm = 0
        for (k, p), (k_base, p_base) in zip(unlearn_model.named_parameters(), self.base_model.named_parameters()):
            # print(k)
            if p.requires_grad:
                l2_loss += (p - p_base).pow(2).sum().cpu()
                norm += p.pow(2).sum().cpu()
        return {"l2_loss": l2_loss.item(), "norm_loss": torch.sqrt(l2_loss/norm).item()}


class ZeroRetrainForgetting(BaseEvaluator):
    """
    Measure the randomness in the model's prediction by comparing them with based model 
    In other words: Measure the change in accuracy on the unlearned data after retraining the model without the data to be unlearned 
    Used for classification neural network only
    """
    def __init__(self, ref_model_name='dummy'):
        super().__init__()
        self.norm = False
        self.ref_model_name = ref_model_name
    
    # Set norm to true if the classification do not return probabilities of each class (0 < class_prob < 1, sum(class_prob) = 1)
    def set_norm(self, value:bool):
        self.norm = value
        return self

    def kl_divergence(self, p, q):
        return sum(p[i] * torch.log2(p[i]/q[i]) for i in range(len(p)))
    
    @torch.no_grad()
    def eval(self, unlearn_model: nn.Module, device='cuda'):
        base_forget_infosrc = self.infosrc[self.ref_model_name]['forget']
        unlearn_forget_infosrc = self.infosrc['unlearned']['forget']
        
        unlearn_forget_prob = unlearn_forget_infosrc.probs
        base_forget_prob = base_forget_infosrc.probs

        avg = 0.5 * (unlearn_forget_prob + base_forget_prob)
        js = 0.5 * (unlearn_forget_prob*torch.log2(unlearn_forget_prob/avg) + base_forget_prob*torch.log2(base_forget_prob/avg))
        js = js.sum(dim=1)
        zrf = js.mean()
        return (1 - zrf).item()

        # for idx in range(unlearn_forget_prob.size(0)):
        #     avg = 0.5 * torch.add(unlearn_forget_prob[idx], base_forget_prob[idx])
        #     js = 0.5 * (self.kl_divergence(unlearn_forget_prob[idx], avg) + self.kl_divergence(base_forget_prob[idx], avg))
        #     zrf += js

        # return (1 - zrf/unlearn_forget_prob.size(0)).item()
