# -*- coding: utf-8 -*- 
# Ciari10 数据集并不能使用 ResNet50 直接运行，原因是ResNet50层数太大，而
#    cifar10分辨率低，在中间卷积层时分辨率就会下降到1，无法执行后续的AveragePool操作
#    如果在ResNet中去除 l3a-l5c 之间的层，就可以拿来训练

from resnet50 import ResNet50
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10

# 导入数据
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
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


# 构建模型
data_format = "channels_last"
EPOCHS = 10

cifar10 = Cifar10()
batch_size = 512
with tf.Graph().as_default():
    # train_data
    X_tf = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y_tf = tf.placeholder(tf.int32, shape=(None, ))

    # build model
    model = ResNet50(data_format, classes=10)

    # train
    logits = model(X_tf, training=True)
    # print("logits: ", logits)
    # print("y_tf:", y_tf)
    correct_prediction = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), y_tf)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_tf, logits=logits))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    # run
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EPOCHS):
            print("\n\nEpoch =", epoch)
            while(True):
                batch_x, batch_y = cifar10.next_batch(batch_size)
                if batch_x == batch_y == None:
                    break
                # print("批量样本:", batch_x.shape, batch_y.shape)
                _, train_loss, train_acc = sess.run([train_op, loss, acc], 
                                feed_dict={X_tf: batch_x, y_tf: batch_y})
                print("Ratio:", int(cifar10.idx/cifar10.total*100), "%,\tTrain_loss=", train_loss, ", Train_acc", train_acc*100, "%")

            # valid
            X_test, y_test = cifar10.test()
            valid_acc, valid_loss = sess.run([acc, loss], 
                                feed_dict={X_tf: X_test, y_tf: y_test})
            print("Valid loss:", valid_loss, ", acc:", valid_acc)
            print("---------")

"""
训练结果 约达到 72% 准确率
"""

