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

    def eval(self, unlearn_model: nn.Module):
        l2_loss = 0
        norm = 0
        for (k, p), (k_base, p_base) in zip(unlearn_model.named_parameters(), self.base_model.named_parameters()):
            # print(k)
            if p.requires_grad:
                l2_loss += (p - p_base).pow(2).sum()
                norm += p.pow(2).sum()
        return {"l2_loss": l2_loss, "norm_loss": torch.sqrt(l2_loss/norm)}


class ZeroRetrainForgetting(RetainBaseEvaluator):
    """
    Measure the randomness in the model's prediction by comparing them with based model 
    In other words: Measure the change in accuracy on the unlearned data after retraining the model without the data to be unlearned 
    Used for classification neural network only
    """
    def __init__(self, forget_set, test_set, base_model: nn.Module):
        super().__init__(forget_set, test_set, base_model)
        self.norm = False
    
    # Set norm to true if the classification do not return probabilities of each class (0 < class_prob < 1, sum(class_prob) = 1)
    def set_norm(self, value:bool):
        self.norm = value
    
    def eval(self, unlearn_model: nn.Module):
        unlearn_forget_prob = unlearn_model(self.forget_set)
        if self.norm:
            unlearn_forget_prob = torch.sigmoid(unlearn_forget_prob)
            unlearn_forget_prob = torch.softmax(unlearn_forget_prob, dim=1)
        
        base_forget_prob = self.base_model(self.forget_set)
        if self.norm:
            base_forget_prob = torch.sigmoid(base_forget_prob)
            base_forget_prob = torch.softmax(base_forget_prob, dim=1)

        zrf = 0
        for idx in range(len(self.forget_set)):
            avg = 0.5 * (unlearn_forget_prob[idx] + base_forget_prob[idx])
            js = 0.5 * (rel_entr(unlearn_forget_prob[idx], avg) + rel_entr(base_forget_prob[idx], avg))
            zrf += js
        
        zrf = zrf / len(self.forget_set)
        return 1 - zrf