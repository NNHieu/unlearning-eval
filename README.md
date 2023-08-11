# Unlearning-Evaluation-Framework (UEF)

## Example

1. Define your unlearning algorithm
```python
def unlearning(net, retain, forget, validation):
    ...
    return unlearned_net
```

2. Setup evaluation settings

```python
# Setup model and dataset for evaluation
data_model_set = Cifar10_Resnet18_Set(data_root='data/cifar10', 
                                    data_plit_RNG=RNG,
                                    index_local_path='data/cifar10/forget_idx.npy',
                                    model_path='data/cifar10/weights_resnet18_cifar10.pth',
                                    download_index=False)
# Init pipeline
pipeline = Pipeline(DEVICE, RNG, data_model_set)

# Prepare evaluators
evaluators = [
      ClassificationAccuracyEvaluator(forget_loader, test_loader, None, None),
      ActivationDistance(forget_loader, test_loader, retrained_model),
      ZeroRetrainForgetting(forget_images, test_images, retrained_model),
      SimpleMiaEval(forget_loader, test_loader, nn.CrossEntropyLoss(reduction="none"), n_splits=10, random_state=0)
  ]
pipeline.set_evaluators(evaluators)
```

3. Run evaluation

```python
print(pipeline.eval(unlearning))
```

example output
```json

{
'ClassificationAccuracyEvaluator': 
    {'forget_set': {'acc': 0.8546, 'f1': 0.8546}, 
     'test_set': {'acc': 0.841, 'f1': 0.841}}, 
'ActivationDistance': 
    {'l2_loss': tensor(663.6556), 'norm_loss': tensor(0.6843)}, 
'ZeroRetrainForgetting': 
     array([0.9925184 , 0.9943087 , 0.9878444 , 0.98459977, 0.990898  ,
       0.9874231 , 0.993134  , 0.9916997 , 0.99533933, 0.9941808 ]), 
'SimpleMiaEval': array([0.512, 0.529, 0.515, 0.505, 0.52 , 0.501, 0.513, 0.504, 0.498,
       0.503])}

```

Please refer to `example.py` for a complete example.

## How to define new evaluation settings

You can define a new class that prepare datasets (original dataset and forget set) along with pretrained model and retrained model.

Refer to `unlearn_eval.pipelines.cifar10_resnet.Cifar10_Resnet18_Set` for an example.

## How to define new evaluators

You can define a new evaluator by inheriting `unlearn_eval.evaluators.base_evaluator.BaseEvaluator` class.


