import copy
import logging
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import os
from utils import accuracy
logger = logging.getLogger('logger')
import wandb
import json
import time
from unlearn_eval.utils import InformationSource


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


class Pipeline():
    def __init__(self, DEVICE, RNG, data_model_set) -> None:
        self.DEVICE = DEVICE
        self.RNG = RNG
        self.data_model_set = data_model_set
    
    def set_evaluators(self, evaluators):
        self._evaluators = evaluators

    def eval(self, unlearning_fn):
        forget_loader, retain_loader, test_loader, val_loader, train_loader = self.data_model_set.get_dataloader(self.RNG)
        # print(next(iter(forget_loader)))
        original_model = self.data_model_set.get_pretrained_model()
        original_model.to(self.DEVICE)
        retrained_model = self.data_model_set.get_retrained_model()
        retrained_model.to(self.DEVICE)
        dummy_model = self.data_model_set.get_init_model()
        dummy_model.to(self.DEVICE)

        # logger.warning("Original model's performance")
        # import IPython
        # IPython.embed()
        # exit(0)
        # logger.warning(f"Retain set accuracy: {accuracy(original_model, retain_loader, self.DEVICE)}")
        # logger.warning(f"Test set accuracy: {accuracy(original_model, test_loader, self.DEVICE)}")
        # logger.warning(f"Forget set accuracy: {accuracy(original_model, forget_loader, self.DEVICE)}")
        # exit(0)

        # logger.warning("Done unlearn")
        start_time = time.time()

        unlearned_model = copy.deepcopy(original_model)
        # Execute the unlearing routine. This might take a few minutes.
        unlearned_model = unlearning_fn(unlearned_model, retain_loader, forget_loader, test_loader, device=self.DEVICE)

        # Prepare the information sources
        unlearn_forget_infosrc = InformationSource(unlearned_model, forget_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        unlearn_test_infosrc = InformationSource(unlearned_model, test_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        unlearn_retain_infosrc = InformationSource(unlearned_model, retain_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        original_forget_infosrc = InformationSource(original_model, forget_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        original_test_infosrc = InformationSource(original_model, test_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        original_retain_infosrc = InformationSource(original_model, retain_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        retrain_forget_infosrc = InformationSource(retrained_model, forget_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        retrain_test_infosrc = InformationSource(retrained_model, test_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        retrain_retain_infosrc = InformationSource(retrained_model, retain_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        dummy_forget_infosrc = InformationSource(dummy_model, forget_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        dummy_test_infosrc = InformationSource(dummy_model, test_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)
        dummy_retain_infosrc = InformationSource(dummy_model, retain_loader, criterion=self.data_model_set.criterion, device=self.DEVICE)

        end_time = time.time()
        logger.warning(f"Unlearning {unlearning_fn.__name__} took {end_time - start_time} seconds")
        # print(f"Retain set accuracy: {100.0 * accuracy(ft_model, retain_loader):0.1f}%")
        # print(f"Test set accuracy: {100.0 * accuracy(ft_model, test_loader):0.1f}%")

        # logger.warning(f"Retain set accuracy: {accuracy(ft_model, retain_loader, self.DEVICE)}")
        # logger.warning(f"Test set accuracy: {accuracy(ft_model, test_loader, self.DEVICE)}")
        # logger.warning(f"Forget set accuracy: {accuracy(ft_model, forget_loader, self.DEVICE)}")

        # return ft_model
        # exit(0)
        # evaluate the unlearned model
        results = {
            "original_model": f"{original_retain_infosrc.accuracy:.4f} / {original_test_infosrc.accuracy:.4f} / {original_forget_infosrc.accuracy:.4f}",
            "unlearned_model": f"{unlearn_retain_infosrc.accuracy:.4f} / {unlearn_test_infosrc.accuracy:.4f} / {unlearn_forget_infosrc.accuracy:.4f}",
            "unlearning_time": f"{end_time - start_time:.4f}",
        }

        infosrc_dict = {
            "original": {
                "forget": original_forget_infosrc,
                "test": original_test_infosrc,
                "retain": original_retain_infosrc
            },
            "unlearned": {
                "forget": unlearn_forget_infosrc,
                "test": unlearn_test_infosrc,
                "retain": unlearn_retain_infosrc
            },
            "retrained": {
                "forget": retrain_forget_infosrc,
                "test": retrain_test_infosrc,
                "retain": retrain_retain_infosrc
            },
            "dummy": {
                "forget": dummy_forget_infosrc,
                "test": dummy_test_infosrc,
                "retain": dummy_retain_infosrc
            }
        }

        for evaluator in self._evaluators:
            # print("Starting evaluation for", evaluator.__class__.__name__)
            evaluator.set_inforsrc(infosrc_dict)
            eval_value = None
            try:
                eval_value = evaluator.eval(unlearned_model, device=self.DEVICE)
                logger.info(f"Evaluation for {evaluator.__class__.__name__} result: {eval_value}")
            except Exception as e:
                logger.error(f"Failed to evaluate {evaluator.__class__.__name__}: {e}")
                eval_value = -1
                raise e
            
            if "LIRA" in evaluator.__class__.__name__:
                results[evaluator.__class__.__name__] = f'{eval_value.get("auroc", -1):.4f}'
            elif "SimpleMiaEval" in evaluator.__class__.__name__:
                results[evaluator.__class__.__name__] = f'{eval_value.mean():.4f}'
            elif "ZeroRetrainForgetting" in evaluator.__class__.__name__:
                results[evaluator.__class__.__name__] = f'{eval_value:.4f}'
            elif "ActivationDistance" in evaluator.__class__.__name__:
                results["L2_Loss"] = f'{eval_value.get("l2_loss", -1):.4f}'
                results["Norm_Loss"] = f'{eval_value.get("norm_loss", -1):.4f}'

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
        
        
        wandb.init(project="machine-unlearning", entity="mtuann", name="Retain_forget_4000_1000")
        # wandb.watch(model, log="all")
        # set name for exp
        # wandb.run.name = f"[5000]"
        best_test_acc = 0
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
            
            train_loss, train_acc, train_sample, _ = accuracy(model, self.retain_loader, self.DEVICE)
            test_loss, test_acc, test_sample, _ = accuracy(model, self.test_loader, self.DEVICE)
            logger.info(f"Epoch: {epoch} / {retain_epochs}, Train loss: {train_loss}, Train acc: {train_acc}, Test loss: {test_loss}, Test acc: {test_acc}")
            tqdm_loader.set_description(f"Epoch: {epoch} / {retain_epochs}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                model_ep_path = f"./models/cifar10_resnet18_4000_1000/weights_ep{epoch}__{train_acc:.4f}__{test_acc:.4f}.pth"
                os.makedirs(os.path.dirname(model_ep_path), exist_ok=True)
                torch.save(model.state_dict(), model_ep_path)
                logger.info(f"Save model to {model_ep_path}")

            # model_ep_path = f"./models/Aug_retain_weights_resnet18_cifar10_ep{epoch}__{train_acc:.4f}__{test_acc:.4f}.pth"
            
            # torch.save(model.state_dict(), model_ep_path)
            # logger.info(f"Save model to {model_ep_path}")
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc}, step= epoch)
        return model

