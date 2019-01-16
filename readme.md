## Resnet50的实现

### 参考：

[1] Deep Residual Learning for Image Recognition(https://arxiv.org/abs/1512.03385).

[2] https://github.com/tensorflow/tensorflow/tree/r1.11/tensorflow/contrib/eager/python/examples/resnet50

### 要点：
1. 模型定义在 `resnet50.py` 中，模型定义为 `tf.keras.Model` 的子类，可以使用 `keras` 中定义的函数，减少代码量。关于这一使用方法参见 `https://www.tensorflow.org/guide/keras`.

2. 由 `tf.keras.Model` 定义的模型类可以分别和 (1) 传统的 `tf.Graph` 以及新发布的 `tf.contrib.eager` 进行结合. 相结合的方法分别定义在 `resnet50_cifar_graph` 和 `resnet50_cifar_eager` 中。

3. 由于使用的 CIFAR10 数据集尺度为`32*32`，而 Resnet50 网络规模较大，在中间层时尺度会缩减为1，不能继续后面的层而报错。因为这里为了演示而把 `ResNet中l3a-l5c 之间的层` 注释掉了。



