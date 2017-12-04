import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, args):
        self.args = args


class SimpleDQNModel(Model):
    def __init__(self, session, actions_count, args):
        # Call super class
        super(SimpleDQNModel, self).__init__(args)

        self.session = session

        # TODO: put that in args
        self.channels = 3
        self.resolution = (40, 40) + (self.channels,)
        self.learning_rate = 0.00025
        self.discount_factor = 0.99

        # Create the input.
        self.s_ = tf.placeholder(shape=[None] + list(self.resolution),
                                 dtype=tf.float32)
        self.q_ = tf.placeholder(shape=[None, actions_count],
                                 dtype=tf.float32)

        # Create the network.
        conv1 = tf.contrib.layers.conv2d(self.s_,
                                         num_outputs=8,
                                         kernel_size=[3, 3],
                                         stride=[2, 2])
        conv2 = tf.contrib.layers.conv2d(conv1,
                                         num_outputs=16,
                                         kernel_size=[3, 3],
                                         stride=[2, 2])
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat,
                                                num_outputs=128)

        self.q = tf.contrib.layers.fully_connected(fc1,
                                                   num_outputs=actions_count,
                                                   activation_fn=None)
        self.action = tf.argmax(self.q, 1)

        self.loss = tf.losses.mean_squared_error(self.q_, self.q)

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def learn(self, state, q):
        state = state.astype(np.float32)
        l, _ = self.session.run([self.loss, self.train_step],
                                feed_dict={self.s_: state, self.q_: q})
        return l

    def get_q(self, state):
        state = state.astype(np.float32)
        return self.session.run(self.q,
                                feed_dict={self.s_: state})

    def get_action(self, state):
        state = state.astype(np.float32)
        state = state.reshape([1] + list(self.resolution))
        return self.session.run(self.action,
                                feed_dict={self.s_: state})[0]