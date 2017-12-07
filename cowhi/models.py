import numpy as np
import tensorflow as tf

import logging
_logger = logging.getLogger(__name__)


class Model(object):
    def __init__(self, args, rng):
        _logger.info("Initializing Model (Type: %s)" %
                     args.model)
        self.args = args
        self.rng = rng


class SimpleDQNModel(Model):
    def __init__(self, args, rng, session, input_shape, output_shape):
        # Call super class
        super(SimpleDQNModel, self).__init__(args, rng)
        self.session = session
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Create the input layer.
        self.s_ = tf.placeholder(shape=[None] + list(self.input_shape),
                                 dtype=tf.float32)
        # Create the hidden layers of the network.
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
        # Create the output layer of the network
        self.q = tf.contrib.layers.fully_connected(fc1,
                                                   num_outputs=self.output_shape,
                                                   activation_fn=None)
        # Define a placeholder for loss calculation
        self.q_ = tf.placeholder(shape=[None, self.output_shape],
                                 dtype=tf.float32)
        # Define important network parameters
        self.loss = tf.losses.mean_squared_error(self.q_, self.q)
        self.optimizer = tf.train.RMSPropOptimizer(self.args.alpha)
        self.train_step = self.optimizer.minimize(self.loss)

        # Define layer for selecting only the max value
        self.action = tf.argmax(self.q, 1)

        # Define layer for a softmax output
        self.action_probs = tf.contrib.layers.softmax(self.q)

    def train(self, state, q):
        state = state.astype(np.float32)
        loss_batch, _ = self.session.run([self.loss, self.train_step],
                                         feed_dict={self.s_: state, self.q_: q})
        return loss_batch

    def get_qs(self, state):
        """ Returns the Q values for all available outputs. """
        state = state.astype(np.float32)
        return self.session.run(self.q,
                                feed_dict={self.s_: state})

    def get_action_probs(self, state):
        """ Returns a probability distribution over the possible actions. """
        state = state.astype(np.float32)
        return self.session.run(self.action_probs,
                                feed_dict={self.s_: state})

    def get_action(self, state):
        """ Returns the index from the maximal Q value """
        state = state.astype(np.float32)
        state = state.reshape([1] + list(self.input_shape))
        return self.session.run(self.action,
                                feed_dict={self.s_: state})[0]
