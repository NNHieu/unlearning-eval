import copy
import logging
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch

from utils import accuracy
logger = logging.getLogger('logger')


def accuracy(net, loader, DEVICE):
    """Return accuracy on a dataset given by the data loader."""
    total_loss, total_sample, total_acc = 0, 0, 0
    net.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total_loss = nn.CrossEntropyLoss()(outputs, targets).item()
        total_sample += targets.size(0)
        total_acc += predicted.eq(targets).sum().item()
    return (total_loss/ total_sample, total_acc / total_sample, total_sample)
    # return correct / total


class Pipeline():
    def __init__(self, DEVICE, RNG, data_model_set) -> None:
        self.DEVICE = DEVICE
        self.RNG = RNG
        self.data_model_set = data_model_set
    
    def set_evaluators(self, evaluators):
        self._evaluators = evaluators

    def eval(self, unlearning_fn):
        forget_loader, retain_loader, test_loader, val_loader = self.data_model_set.get_dataloader(self.RNG)
        # print(next(iter(forget_loader)))
        original_model = self.data_model_set.get_pretrained_model()
        original_model.to(self.DEVICE)
        
        logger.warning("Original model's performance")
        # import IPython
        # IPython.embed()
        # exit(0)
        logger.warning(f"Retain set accuracy: {accuracy(original_model, retain_loader, self.DEVICE)}")
        logger.warning(f"Test set accuracy: {accuracy(original_model, test_loader, self.DEVICE)}")
        logger.warning(f"Forget set accuracy: {accuracy(original_model, forget_loader, self.DEVICE)}")
        # exit(0)
        logger.warning("Done unlearn")
        ft_model = copy.deepcopy(original_model)
        # Execute the unlearing routine. This might take a few minutes.
        ft_model = unlearning_fn(ft_model, retain_loader, forget_loader, test_loader)

        # print(f"Retain set accuracy: {100.0 * accuracy(ft_model, retain_loader):0.1f}%")
        # print(f"Test set accuracy: {100.0 * accuracy(ft_model, test_loader):0.1f}%")
        logger.warning(f"Retain set accuracy: {accuracy(ft_model, retain_loader, self.DEVICE)}")
        logger.warning(f"Test set accuracy: {accuracy(ft_model, test_loader, self.DEVICE)}")
        logger.warning(f"Forget set accuracy: {accuracy(ft_model, forget_loader, self.DEVICE)}")
        
        # exit(0)
        # evaluate the unlearned model
        results = {}
        for evaluator in self._evaluators:
            results[evaluator.__class__.__name__] = evaluator.eval(ft_model, device=self.DEVICE)
        
        return results
    
    def training_retain_from_scratch(self, retain_epochs=100):
        # train from scratch
        logger.info(f"Retrain retrain dataset from scratch")
        model = self.data_model_set.get_init_model().to(self.DEVICE)
        
        self.retain_loader = self.data_model_set.retain_loader
        self.test_loader = self.data_model_set.test_loader
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
        
        
        
        for epoch in range(retain_epochs):
            model.train()
            tqdm_loader = tqdm.tqdm(self.retain_loader, desc="Retrain from scratch")    
            for inputs, targets in tqdm_loader:
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            train_loss, train_acc, train_sample = accuracy(model, self.retain_loader, self.DEVICE)
            test_loss, test_acc, test_sample = accuracy(model, self.test_loader, self.DEVICE)
            logger.info(f"Epoch: {epoch} / {retain_epochs}, Train loss: {train_loss}, Train acc: {train_acc}, Test loss: {test_loss}, Test acc: {test_acc}")
            tqdm_loader.set_description(f"Epoch: {epoch} / {retain_epochs}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
            
            model_ep_path = f"./models/Aug_retain_weights_resnet18_cifar10_ep{epoch}__{train_acc:.4f}__{test_acc:.4f}.pth"
            
            torch.save(model.state_dict(), model_ep_path)
            logger.info(f"Save model to {model_ep_path}")
        return model

