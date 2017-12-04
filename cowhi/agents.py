from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf

from cowhi.models import SimpleDQNModel
from cowhi.replaymemories import SimpleReplayMemory

import logging
_logger = logging.getLogger(__name__)


class Agent(object):
    def __init__(self, env, args, target_dir=None):
        self.args = args
        self.env = env
        self.rewards = 0.0
        self.observation = None
        self.model_path = os.path.join(target_dir, 'models')
        self.start_time = time.time()

    def print_stats(self, elapsed_time, step, step_num, train_scores):
        steps_per_s = 1.0 * step / elapsed_time
        steps_per_m = 60.0 * step / elapsed_time
        steps_per_h = 3600.0 * step / elapsed_time
        steps_remain = step_num - step
        remain_h = int(steps_remain / steps_per_h)
        remain_m = int((steps_remain - remain_h * steps_per_h) / steps_per_m)
        remain_s = int((steps_remain - remain_h * steps_per_h - remain_m * steps_per_m) / steps_per_s)
        elapsed_h = int(elapsed_time / 3600)
        elapsed_m = int((elapsed_time - elapsed_h * 3600) / 60)
        elapsed_s = int((elapsed_time - elapsed_h * 3600 - elapsed_m * 60))
        print("{}% | Steps: {}/{}, {:.2f}M step/h, {:02}:{:02}:{:02}/{:02}:{:02}:{:02}".format(
            100.0 * step / step_num, step, step_num, steps_per_h / 1e6,
            elapsed_h, elapsed_m, elapsed_s, remain_h, remain_m, remain_s), file=sys.stderr)

        mean_train = 0
        std_train = 0
        min_train = 0
        max_train = 0
        if (len(train_scores) > 0):
            train_scores = np.array(train_scores)
            mean_train = train_scores.mean()
            std_train = train_scores.std()
            min_train = train_scores.min()
            max_train = train_scores.max()
        print("Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
            len(train_scores), mean_train, std_train, min_train, max_train), file=sys.stderr)

class SingleTaskAgent(Agent):
    def __init__(self, args):
        # Call super class
        super(SingleTaskAgent, self).__init__(args)


class RandomAgent(Agent):
    """Simple agent for DeepMind Lab."""

    def _action(*entries):
        return np.array(entries, dtype=np.intc)

    ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
        'look_up': _action(0, 10, 0, 0, 0, 0, 0),
        'look_down': _action(0, -10, 0, 0, 0, 0, 0),
        'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
        'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, 1, 0, 0, 0),
        'backward': _action(0, 0, 0, -1, 0, 0, 0),
        'fire': _action(0, 0, 0, 0, 1, 0, 0),
        'jump': _action(0, 0, 0, 0, 0, 1, 0),
        'crouch': _action(0, 0, 0, 0, 0, 0, 1)
    }

    ACTION_LIST = ACTIONS.values()

    def __init__(self, args):
        # Call super class
        super(RandomAgent, self).__init__(args)
        print('Starting random discretized agent.')
        self.reset()

    def step(self):
        """Gets an image state and a reward, returns an action."""
        return random.choice(self.ACTION_LIST)

    def reset(self):
        self.rewards = 0.0
        self.observation = None


class SimpleDQNAgent(Agent):
    def __init__(self, env, args, target_dir):
        # Call super class
        super(SimpleDQNAgent, self).__init__(env, args, target_dir)

        # TODO: put that in args
        self.start_eps = 1.0
        self.end_eps = 0.1
        self.eps_decay_iter = 0.33 * self.args.length
        self.model_backup_frequency = 0.01 * self.args.length
        self.channels = 3
        self.resolution = (40, 40) + (self.channels,)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)

        self.model = SimpleDQNModel(self.session, self.env.num_actions, self.args)
        self.memory = SimpleReplayMemory(self.args)

        self.rewards = 0

        self.saver = tf.train.Saver(max_to_keep=1000)
        if self.args.load_model is not None:
            # print("Loading model from: ", self.args.load_model)
            self.saver.restore(self.session, self.args.load_model)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)

    def train_model(self):
        # train model with random batch from memory
        if self.memory.size > 2 * self.memory.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_batch(self.memory.batch_size)

            q = self.model.get_q(s1)
            q2 = np.max(self.model.get_q(s2), axis=1)
            q[np.arange(q.shape[0]), a] = r + (1 - isterminal) * self.model.discount_factor * q2
            self.model.learn(s1, q)

    def get_action(self, state):
        if random.random() <= 0.05:
            a = random.randint(0, self.env.num_actions-1)
        else:
            a = self.model.get_action(state)
        return a

    def step(self, iteration):
        s = self.preprocess_observation(self.env.get_observation())

        # Epsilon-greedy.
        if iteration < self.eps_decay_iter:
            eps = self.start_eps - iteration / self.eps_decay_iter * (self.start_eps - self.end_eps)
        else:
            eps = self.end_eps

        if random.random() <= eps:
            a = random.randint(0, self.env.num_actions-1)
        else:
            a = self.model.get_action(s)

        reward = self.env.step(a)
        self.rewards += reward

        is_terminal = not self.env.is_running()
        self.memory.add(s, a, is_terminal, reward)
        self.train_model()

    def train(self):
        print("Starting training.")
        train_scores = []
        self.env.reset()
        for step in xrange(1, self.args.length+1):
            self.step(step)
            if not self.env.is_running():
                train_scores.append(self.rewards)
                self.rewards = 0
                self.env.reset()

            if step % self.model_backup_frequency == 0:
                model_name_curr = os.path.join(
                    self.model_path,
                    "DQN_{:04}".format(int(step / self.model_backup_frequency)))
                print("\nSaving the network weights to:",
                      model_name_curr,
                      file=sys.stderr)
                self.saver.save(self.session, model_name_curr)

                # TODO: Update logger to logging
                self.print_stats(time.time() - self.start_time,
                                 step,
                                 self.args.length,
                                 train_scores)

                train_scores = []

        self.env.reset()

    def preprocess_observation(self, img):
        if self.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.resolution[1], self.resolution[0]))
        return np.reshape(img, self.resolution)

