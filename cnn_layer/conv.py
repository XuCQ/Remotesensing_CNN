# coding=utf-8
import numpy as np
import tensorflow as tf


class ConvLayer(object):
    def __init__(self, input_shape, n_size, n_filter, stride=1, activation=tf.nn.relu,
                 batch_normal=False, weight_decay=None, name='conv'):
        """
        :param input_shape: the shape of input pic
        :param n_size: conv size
        :param n_filter: filter num
        :param stride:
        :param activation:
        :param batch_normal:
        :param weight_decay:
        :param name: cnn_layer name
        """
        self.input_shape = input_shape
        self.n_filter = n_filter
        self.stride = stride
        self.activation = activation
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay

        # weight_variable
        self.weight = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[n_size, n_size, self.input_shape[3], self.n_filter],
                mean=0.0,
                stddev=np.sqrt(2.0 / (self.input_shape[1] * self.input_shape[2] * self.input_shape[3]))),
            # input[0] is baych size
            name='W_%s' % (name))

        # weight decay, L2正则化 lambda*||w||2
        if self.weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(self.weight), self.weight_decay)  # tf.nn.l2_loss :1/2Σw²
            tf.add_to_collection('losses', weight_decay)

        # bias_variable
        self.bias = tf.Variable(
            initial_value=tf.constant(
                0.0, shape=[self.n_filter]),
            name='b_%s' % (name))

        # batch normalization
        if self.batch_normal:
            self.epsilon = 1e-5
            self.gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[self.n_filter]), name='gamma_%s' % (name))

    def get_output(self, input):
        # calculate output_shape
        self.output_shape = [self.input_shape[0], int(self.input_shape[1] / self.stride),
                             int(self.input_shape[2] / self.stride), self.n_filter]

        # conv cnn_layer
        self.conv = tf.nn.conv2d(
            input=input,
            filter=self.weight,
            strides=[1, self.stride, self.stride, 1],
            padding='SAME'
        )

        # batch normalization
        if self.batch_normal:
            mean, variance = tf.nn.moments(self.conv, axes=[0, 1, 2], keep_dims=False)
            # output_size: 64, just need size of n_filter，对于一个batch求均值和标准差，最终输出一个list，大小为卷继层数
            # axis = list(range(len(self.conv.get_shape()) - 1))
            self.hidden = tf.nn.batch_normalization(self.conv, mean, variance, self.bias, self.gamma, self.epsilon)
            # gamma*x+bias, 这两个参数均需要通过网络训练获得；epsilon设置是在normalization时防止分母为0
        else:
            self.hidden = self.conv + self.bias

        # activation
        if self.activation:
            self.output=self.activation(self.hidden)
        else:
            self.output=self.hidden

        return self.output

