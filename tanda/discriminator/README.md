# TANDA Discriminators

## Note about using batch norm:

Models should use `tf.contrib.layers.batch_norm` and set as follows:
```python
tf.contrib.layers.batch_norm(
      x,
      decay=0.9,
      center=True, # I.e. use beta
      scale=True, # I.e. use gamma
      epsilon=1e-5,
      # Note: important to leave this unset!
      # updates_collections=None,
      variables_collections=[self.bn_vars_collection],
      is_training=train,
      reuse=reuse,
      scope=name,
      trainable=True
    )
```
where the important ones are:
- **Specifically leave `updates_collections` unset (_do not set as `None`_)**. 
- Set `variables_collections=[self.bn_vars_collection]` if you want to be able to save the model properly
- Pass though `train`, `reuse`, and `name` vars into operator

## Models
* `DCNN` (`dcnn.py`): **Current default TAN discriminator for images.** Deep All-Convolutional Net from [DCGAN paper](https://arxiv.org/abs/1511.06434)

* `ResNetDefault` (`resnet_cifar.py`): **Current default end model for CIFAR10.** ResNet adapted for CIFAR10/100 from [ResNet paper](https://arxiv.org/abs/1512.03385), code from [tensorflow/models](https://github.com/tensorflow/models/tree/master/resnet)
