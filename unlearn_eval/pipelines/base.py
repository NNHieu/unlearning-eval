import copy

class Pipeline():
    def __init__(self, DEVICE, RNG, data_model_set) -> None:
        self.DEVICE = DEVICE
        self.RNG = RNG
        self.data_model_set = data_model_set
    
    def set_evaluators(self, evaluators):
        self._evaluators = evaluators

    def eval(self, unlearning_fn):
        forget_loader, retain_loader, test_loader, val_loader = self.data_model_set.get_dataloader(self.RNG)
        original_model = self.data_model_set.get_pretrained_model()
        original_model.to(self.DEVICE)
        # print("Original model's performance")
        # print(f"Retain set accuracy: {100.0 * accuracy(original_model, retain_loader):0.1f}%")
        # print(f"Test set accuracy: {100.0 * accuracy(original_model, test_loader):0.1f}%")

        ft_model = copy.deepcopy(original_model)
        # Execute the unlearing routine. This might take a few minutes.
        ft_model = unlearning_fn(ft_model, retain_loader, forget_loader, test_loader)

        # print(f"Retain set accuracy: {100.0 * accuracy(ft_model, retain_loader):0.1f}%")
        # print(f"Test set accuracy: {100.0 * accuracy(ft_model, test_loader):0.1f}%")

        # evaluate the unlearned model
        results = {}
        for evaluator in self._evaluators:
            results[evaluator.__class__.__name__] = evaluator.eval(ft_model, device=self.DEVICE)
        
        return results