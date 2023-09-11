import torch
from sklearn import linear_model, model_selection
import numpy as np
from .base import BaseEvaluator
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.data import Subset, DataLoader
from torchvision.models import resnet18
from .base import BaseEvaluator
from sklearn.metrics import roc_auc_score


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

def _simple_mia_2(sample_loss, members, forget_losses):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    attack_model.fit(sample_loss, members)
    # cv = model_selection.StratifiedShuffleSplit(
    #     n_splits=n_splits, random_state=random_state
    # )
    # return model_selection.cross_val_score(
    #     attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    # )
    return attack_model.predict(forget_losses).mean()

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
        retain_infosrc = self.infosrc['unlearned']['retain']
        forget_infosrc = self.infosrc['unlearned']['forget']
        test_infosrc = self.infosrc['unlearned']['test']

        retain_losses = retain_infosrc.losses
        test_losses = test_infosrc.losses
        samples_mia = np.concatenate((test_losses, retain_losses)).reshape((-1, 1))
        labels_mia = [0] * len(test_losses) + [1] * len(retain_losses)
        forget_losses = forget_infosrc.losses.numpy().reshape(-1, 1)
        # mia_scores = _simple_mia(samples_mia, labels_mia, n_splits=self.n_splits, random_state=self.random_state)
        mia_scores = _simple_mia_2(samples_mia, labels_mia, forget_losses)
        return np.array([mia_scores])
    

class LIRA(BaseEvaluator):

    def __init__(self, forget_set, test_set, forget_idx, train_set, noaug_train_set, num_exps, pkeep=0.5, device='cuda'):
        super().__init__(forget_set, test_set)
        self.forget_idx = forget_idx
        self.train_set = train_set
        self.noaug_train_set = noaug_train_set
        self.noaug_train_loader = DataLoader(noaug_train_set, batch_size=32, num_workers=4, pin_memory=True, shuffle=False)
        self.labels = torch.tensor(train_set.targets, device=device)
        self.num_exps = num_exps
        self.pkeep = pkeep
        self.device = device
        if not os.path.exists('exp/lira/'):
            os.makedirs('exp/lira', exist_ok=True)
            self.split_train_data()
        else:
            self.keep = np.load('exp/lira/keep.npy')
        if not os.path.exists('exp/lira/shadow_model'):
            os.makedirs('exp/lira/shadow_model', exist_ok=True)
            self.model_list, self.score_list = self.train_shadow_model()
        else:
            self.score_list = self.load_shadow_model()

    def split_train_data(self):
        np.random.seed(0)
        keep = np.random.uniform(0,1,size=(self.num_exps, len(self.train_set)))
        order = keep.argsort(0)
        keep = order < int(self.pkeep * self.num_exps)
        self.keep = keep
        np.save('exp/lira/keep.npy', keep)
        # keep = np.array(keep[FLAGS.expid], dtype=bool)

    def get_data(self, expid):
        keep = self.keep[expid].nonzero()[0]
        dset = Subset(self.train_set, keep)
        loader = DataLoader(dset, batch_size=128, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
        return loader
    
    def create_model(self):
        return resnet18(num_classes=10).to(self.device)
    
    def train(self, model, train_loader):
        n_epochs = 100
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], 0.1)
        model.train()
        for i in trange(n_epochs):
            for x, label in train_loader:
                x, label = x.to(self.device), label.to(self.device)
                logit = model(x)
                loss = F.cross_entropy(logit, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        # acc = self.validate(model)
        # print(acc)
        logits = self.get_logits(model, self.noaug_train_loader)
        scores = self.get_score(logits, self.labels)
        return model, scores

    @torch.no_grad()
    def get_logits(self, model, loader):
        model.eval()
        logits = []
        for x, label in loader:
            x = x.to(self.device)
            logits.append(model(x))
        return torch.cat(logits)

    @torch.no_grad()    
    def validate(self, model):
        model.eval()
        corr, total = 0, 0
        for x, label in self.test_set:
            x, label = x.to(self.device), label.to(self.device)
            pred = model(x).argmax(1)
            corr += (pred == label).sum()
            total += label.shape[0]
        return corr / total
    
    def train_shadow_model(self):
        model_list, score_list = [], []
        for expid in range(self.num_exps):
            loader = self.get_data(expid)
            model = self.create_model()
            model, score = self.train(model, loader)
            torch.save(model.state_dict(), f'exp/lira/shadow_model/resnet18_cifar10_{expid}.pth')
            torch.save(score, f'exp/lira/shadow_model/resnet18_cifar10_score_{expid}.pth')
            model_list.append(model)
            score_list.append(score)
        return model_list, score_list

    def load_shadow_model(self, load_model=False):
        if load_model:
            model_list = []
        logits_list = []
        for expid in range(self.num_exps):
            if load_model:
                model = self.create_model()
                model.load_state_dict(torch.load(f'exp/lira/shadow_model/resnet18_cifar10_{expid}.pth'))
                model_list.append(model)
            logits_list.append(torch.load(f'exp/lira/shadow_model/resnet18_cifar10_score_{expid}.pth'))
        if load_model:
            return model_list, logits_list
        return logits_list

    def get_score(self, logits, labels):
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        y_true = logits[torch.arange(len(logits)), labels]
        logits[:, labels] = 0
        y_wrong = torch.logsumexp(logits, dim=1)
        return y_true - y_wrong
    
    def compute_lira(self, score, dat_out, fix_variance=True):
        try:
            mean_out = torch.mean(dat_out, dim=1)
        except:
            mean_out = torch.mean(dat_out)
        if fix_variance:
            std_out = torch.std(dat_out)
        else:
            std_out = torch.std(dat_out, dim=1)
        out_rv = torch.distributions.normal.Normal(mean_out, std_out)
        return out_rv.log_prob(score)

    
    @torch.no_grad()
    def eval(self, unlearn_model: nn.Module, device='cuda'):
        forget_labels = torch.cat([i[1].to(device) for i in self.forget_set])
        test_labels = torch.cat([i[1].to(device) for i in self.test_set])
        forget_predict = self.get_logits(unlearn_model, self.forget_set)
        test_predict = self.get_logits(unlearn_model, self.test_set)
        forget_score = self.get_score(forget_predict, forget_labels)
        test_score = self.get_score(test_predict, test_labels)
        forget_out = []
        scores = torch.stack(self.score_list) # num_exps x train_size
        for i in self.forget_idx:
            forget_out.append(scores[~self.keep[:, i], i])
        out_size = min(map(len, forget_out))
        forget_out = torch.stack([x[:out_size] for x in forget_out])
        forget_lira = self.compute_lira(forget_score, forget_out)
        test_lira = self.compute_lira(test_score, scores.flatten())
        # convert to numpy
        forget_lira = forget_lira.cpu().numpy()
        test_lira = test_lira.cpu().numpy()
        y_true = np.concatenate([np.zeros_like(forget_lira), np.ones_like(test_lira)])
        y_score = np.concatenate([forget_lira, test_lira])
        auroc = roc_auc_score(y_true, y_score)
        return {'forget_lira': forget_lira, 'test_lira': test_lira, 'auroc': auroc}