import torch.nn as nn


class BaseEvaluator:
    forget_set = None
    test_set = None

    def __init__(self):
        pass
    
    def set_inforsrc(self, infosrc_dict):
        # {
        #     "original": {
        #         "forget": original_forget_infosrc,
        #         "test": original_test_infosrc,
        #         "retain": original_retain_infosrc
        #     },
        #     "unlearned": {
        #         "forget": unlearn_forget_infosrc,
        #         "test": unlearn_test_infosrc,
        #         "retain": unlearn_retain_infosrc
        #     }
        # }
        self.infosrc = infosrc_dict

    def eval(self, unlearn_model):
        raise NotImplementedError("Need to inherit")


class RetainBaseEvaluator(BaseEvaluator):
    """
    Retain-model based Evaluator
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def eval(self, unlearn_model: nn.Module):
        raise NotImplementedError("Need to fill up")
