# -*- coding: utf-8 -*- 
# Ciari10 数据集并不能使用 ResNet50 直接运行，原因是ResNet50层数太大，而
#    cifar10分辨率低，在中间卷积层时分辨率就会下降到1，无法执行后续的AveragePool操作
#    如果在ResNet中去除 l3a-l5c 之间的层，就可以拿来训练

from resnet50 import ResNet50
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

# 导入数据
class Cifar10:
    def __init__(self):
        (X_train, y_tr), (X_test, y_te) = cifar10.load_data()
        y_tr = y_tr.reshape(y_tr.shape[0])
        y_te = y_te.reshape(y_te.shape[0])
        #
        X_train = X_train.astype(np.float32) / 255.
        X_test = X_test.astype(np.float32) / 255.
        y_train = y_tr.astype(np.int32)
        y_test = y_te.astype(np.int32)

        print("训练样本数:", X_train.shape[0], ", 测试样本数:", X_test.shape[0])
        self.X_train, self.y_train, self.X_test, self.y_test = self.shuffle(
            X_train, y_train, X_test, y_test)
        self.idx = 0
        self.total = self.X_train.shape[0]

    def shuffle(self, X_train, y_train, X_test, y_test):
        x1 = np.arange(0, X_train.shape[0])
        x2 = np.arange(0, X_test.shape[0])
        np.random.shuffle(x1)
        np.random.shuffle(x2)
        X_train, y_train = X_train[x1], y_train[x1]
        X_test, y_test = X_test[x2], y_test[x2]
        return X_train, y_train, X_test, y_test

    def next_batch(self, batch_size=100):
        if self.idx + batch_size >= self.total:
            self.idx = 0
            return None, None
        batch_x = self.X_train[self.idx: self.idx+batch_size]
        batch_y = self.y_train[self.idx: self.idx+batch_size]
        self.idx += batch_size
        assert batch_x.shape == (batch_size, 32, 32, 3)
        # print("idx =", self.idx)
        return batch_x, batch_y

    def test(self):
        return self.X_test, self.y_test

# data
cifar10 = Cifar10()
valid_x, valid_y = cifar10.test()
valid_x, valid_y = tf.convert_to_tensor(valid_x), tf.convert_to_tensor(valid_y)
batch_size = 512

# build model
data_format = "channels_last"
model = ResNet50(data_format, classes=10)
optimizer = tf.train.AdamOptimizer()

#
EPOCHS = 10
for i in range(EPOCHS):
    print("EPOCHS:", i)
    while(True):
        batch_x, batch_y = cifar10.next_batch(batch_size)
        if batch_x == batch_y == None:
            break
        batch_x, batch_y = tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)
        # train
        with tf.GradientTape() as tape:
            logits = model(batch_x)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_y, logits=logits))
        # train
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))    
        #
        train_prediction = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), batch_y)
        train_acc = tf.reduce_mean(tf.cast(train_prediction, tf.float32))    
        print("Ratio:", int(cifar10.idx/cifar10.total*100), "%,\tTrain_loss=", loss.numpy(), ", Train_acc", train_acc.numpy()*100, "%")

    # # valid
    valid_logits = model(valid_x)
    valid_prediction = tf.equal(tf.argmax(valid_logits, axis=1, output_type=tf.int32), valid_y)
    valid_acc = tf.reduce_mean(tf.cast(valid_prediction, tf.float32))
    valid_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=valid_y, logits=valid_logits))

    print("Average valid loss:", valid_loss.numpy(), ", acc:", valid_acc.numpy())
