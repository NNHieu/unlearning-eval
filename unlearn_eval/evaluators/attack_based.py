import torch
from sklearn import linear_model, model_selection
import numpy as np
from .base import BaseEvaluator


def compute_losses(net, loader, criterion, device):
    """Auxiliary function to compute per-sample losses"""

    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def _simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def simple_mia(model, criterion, forget_loader, test_loader, device, *, n_splits=10, random_state=0):
    with torch.no_grad():
        forget_losses = compute_losses(model, forget_loader, criterion, device=device)
        test_losses = compute_losses(model, test_loader, criterion, device=device)
    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    mia_scores = _simple_mia(samples_mia, labels_mia, n_splits=n_splits, random_state=random_state)
    return mia_scores

class SimpleMiaEval(BaseEvaluator):
    """
    Simple MIA Evaluation
    """
    def __init__(self, n_splits=10, random_state=0):
        self.n_splits = n_splits
        self.random_state = random_state

    def eval(self, unlearn_model: torch.nn.Module, *, device, **kwargs):
        forget_infosrc = self.infosrc['unlearned']['forget']
        test_infosrc = self.infosrc['unlearned']['test']

        forget_losses = forget_infosrc.losses
        test_losses = test_infosrc.losses
        samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
        labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
        mia_scores = _simple_mia(samples_mia, labels_mia, n_splits=self.n_splits, random_state=self.random_state)
        return mia_scores