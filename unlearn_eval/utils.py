import torch
import numpy as np
from tqdm import tqdm

def accuracy(net, loader, DEVICE, mis=True):
    """Return accuracy on a dataset given by the data loader."""
    total_loss, total_sample, total_acc = 0, 0, 0
    net.eval()
    miss_classified = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total_loss = nn.CrossEntropyLoss()(outputs, targets).item()
        total_sample += targets.size(0)
        total_acc += predicted.eq(targets).sum().item()
        # adding labels of missclassified samples
        if mis:
            miss_classified.extend([targets[i].item() for i in range(len(targets)) if predicted[i] != targets[i]])
    
    dict_miss_classified = {}
    if mis:
        for label in miss_classified:
            if label not in dict_miss_classified:
                dict_miss_classified[label] = 1
            else:
                dict_miss_classified[label] += 1
    # print(dict_miss_classified)

    return (total_loss/ total_sample, total_acc / total_sample, total_sample, dict_miss_classified)
    # return correct / total

class InformationSource:
    def __init__(self, model, dataloader, criterion=None, device="cuda") -> None:
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self._info = {}
        self.device = device

    @torch.no_grad()
    def _run(self):
        """Auxiliary function to compute per-sample losses"""

        all_losses = []
        all_logits = []
        all_labels = []
        total_loss, total_sample, total_acc = 0, 0, 0
        for inputs, targets in tqdm(self.dataloader, desc="Computing information source"):
            all_labels.append(targets)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            _, predicted = logits.max(dim=1)
            total_acc += predicted.eq(targets).sum().item()
            total_sample += targets.size(0)

            all_logits.append(logits)
            if self.criterion is not None:
                losses = self.criterion(logits, targets)
                all_losses.append(losses)

        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        if len(all_losses) > 0:
            all_losses = torch.cat(all_losses)
            self._info['loss'] = all_losses.cpu()

        self._info['labels'] = all_labels.cpu()
        self._info['logits'] = all_logits.cpu()
        self._info['prob'] = torch.softmax(all_logits, dim=1).cpu()
        self._info['acc'] = total_acc / total_sample

    @property
    def losses(self):
        if self.criterion is None:
            raise ValueError("No criterion provided")
        if 'loss' not in self._info:
            self._run()
        return self._info['loss']
    
    @property
    def logits(self):
        if 'logits' not in self._info:
            self._run()
        return self._info['logits']

    @property
    def probs(self):
        if 'prob' not in self._info:
            self._run()
        return self._info['prob']

    @property
    def accuracy(self):
        if 'acc' not in self._info:
            self._run()
        return self._info['acc']

    @property
    def labels(self):
        if 'labels' not in self._info:
            self._run()
        return self._info['labels']
    
    @property
    def predictions(self):
        return self.logits.argmax(dim=1)
    
    def reset(self):
        self._info = {}