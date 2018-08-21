import numpy as np
import tensorflow as tf


class DenseLayer(object):
    def __init__(self, input_shape, hidden_dim, activation=None, dropout=False,
                 keep_prob=None, batch_normal=False, weight_decay=None, name='dense'):
        self.batch_size = input_shape[0]
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay
        self.keep_prob = keep_prob

        # weight :n*k
        self.weight = tf.Variable(
            initial_value=tf.random_normal(
                shape=[input_shape[1], self.hidden_dim],
                mean=0.0,
                stddev=np.sqrt(2.0 / input_shape[1])),
            name='W_%s' % (name)
        )

        # weight_decay
        if weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(self.weight), weight_decay)
            tf.add_to_collection('losses', weight_decay)

        # bias
        self.bias = tf.Variable(
            initial_value=tf.constant(
                0.0,
                shape=[self.hidden_dim]),
            name='b_%s' % (name)
        )

        # BN
        if self.batch_normal:
            self.epsilon = 1e-5
            self.gamma = tf.Variable(
                initial_value=tf.constant(1.0, shape=[self.hidden_dim]),
                name='gamma_%s' % (name)
            )

    def getshape(self, input):
        self.output_shape = [self.batch_size, self.hidden_dim]

        # hidden
        intermediate = tf.matmul(input, self.weight)

        # BN
        if self.batch_normal:
            mean, variance = tf.nn.moments(intermediate, axes=[0])
            self.hidden=tf.nn.batch_normalization(intermediate,mean,variance,self.bias,self.gamma,self.epsilon) # scale*x+offset
        else:
            self.hidden=intermediate+self.bias

        # dropout
        if self.dropout:
            self.hidden=tf.nn.dropout(self.hidden,keep_prob=self.keep_prob)

        # activation
        if self.activation:
            self.output=self.activation(self.hidden)
        else:
            self.output=self.hidden

        return self.output