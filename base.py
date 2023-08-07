class BaseEvaluator:
    forget_set = None
    test_set = None

    def __init__(self, forget_set, test_set):
        self.forget_set = forget_set
        self.test_set = test_set

    def eval(self, unlearn_model):
        raise NotImplementedError("Need to inherit")


class AccuracyEvaluator(BaseEvaluator):
    def __init__(self, forget_set, test_set, forget_label, test_label):
        super.__init__(forget_set, test_set)
        self.forget_label = forget_label
        self.test_label = test_label

    def eval(self, unlearn_model):
        raise NotImplementedError("Need to fill up")


class RetainBaseEvaluator(BaseEvaluator):
    def __init__(self, forget_set, test_set, base_model):
        super.__init__(forget_set, test_set)
        self.base_model = base_model

    def eval(self, unlearn_model):
        raise NotImplementedError("Need to fill up")