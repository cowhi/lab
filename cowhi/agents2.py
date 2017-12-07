from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import random
import six
import sys
import time
import numpy as np
import tensorflow as tf

from six.moves import range

from cowhi.models import SimpleDQNModel
from cowhi.replaymemories import SimpleReplayMemory
from cowhi.helper import print_stats

import logging
_logger = logging.getLogger(__name__)


class Agent(object):
    def __init__(self, args, rng, env, target_dir=None):
        _logger.info("Initializing Agent (type: %s, load_model: %s)" %
                     (args.agent, str(isinstance(args.load_model, six.string_types))))
        self.args = args
        self.rng = rng
        self.env = env
        self.rewards = 0.0
        self.epsilon = self.args.epsilon_start


        self.observation = None
        self.model = None
        self.model_name = None
        self.model_last = None
        self.model_input_shape = None
        self.model_path = os.path.join(target_dir, 'models')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.video_path = os.path.join(target_dir, 'videos')
        if not os.path.isdir(self.video_path):
            os.makedirs(self.video_path)
        self.start_time = time.time()

    def preprocess_input(self, img):
        if self.args.color_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.args.input_width, self.args.input_height))
        return np.reshape(img,
                          self.model_input_shape)

    def play(self):
        out_video = None
        video_path = None
        if self.args.save_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_path = os.path.join(
                self.video_path,
                "video_" + self.model_name + ".avi")
            out_video = cv2.VideoWriter(video_path, fourcc, self.args.fps, (self.args.width, self.args.height))

        reward_total = 0
        num_episodes = 1
        while num_episodes != 0:
            if not self.env.is_running():
                self.env.reset()
                print("Total reward: {}".format(reward_total))
                reward_total = 0
                num_episodes -= 1

            state_raw = self.env.get_observation()
            state = self.preprocess_input(state_raw)
            action = self.get_action(state)

            for _ in range(self.args.frame_repeat):
                if self.args.show:
                    cv2.imshow("frame-test", state_raw)
                    cv2.waitKey(20)
                if self.args.save_video:
                    out_video.write(state_raw.astype('uint8'))
                reward = self.env.step(action, 1)
                reward_total += reward
                if not self.env.is_running():
                    break
                state_raw = self.env.get_observation()
        if self.args.save_video:
            out_video.release()
            print("Saved video (fps:%i, size:%s) to: %s" %
                  (self.args.fps, str((self.args.width, self.args.height)), video_path))
        if self.args.show:
            cv2.destroyAllWindows()


class SimpleDQNAgent(Agent):
    def __init__(self, args, rng, env, target_dir):
        # Call super class
        super(SimpleDQNAgent, self).__init__(args, rng, env, target_dir)

        # Prepare model
        tf.set_random_seed(self.args.random_seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True
        # Initiate tensorflow session
        self.session = tf.Session(config=config)
        self.model_input_shape = (self.args.input_width, self.args.input_height) + \
                                 (self.args.color_channels,)
        self.model = SimpleDQNModel(self.args,
                                    self.rng,
                                    self.session,
                                    self.model_input_shape,
                                    self.env.num_actions)
        self.memory = SimpleReplayMemory(self.args,
                                         self.rng,
                                         self.model_input_shape)

        self.rewards = 0

        self.saver = tf.train.Saver(max_to_keep=1000)
        if self.args.load_model is not None:
            self.saver.restore(self.session, self.args.load_model)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)
        self.model_name = "DQN_0000"
        self.model_last = os.path.join(self.model_path, self.model_name)
        self.saver.save(self.session, self.model_last)

    def train_model(self):
        # train model with random batch from memory
        if self.memory.size > 2 * self.args.batch_size:
            s, a, r, s_prime, is_terminal = self.memory.get_batch()
            qs = self.model.get_qs(s)
            max_qs = np.max(self.model.get_qs(s_prime), axis=1)
            qs[np.arange(qs.shape[0]), a] = r + (1 - is_terminal) * self.args.gamma * max_qs
            self.model.train(s, qs)

    def update_epsilon(self, steps):
        # Update epsilon if necessary
        if steps > self.args.epsilon_decay_steps:
            return self.args.epsilon_min
        else:
            return self.args.epsilon_start - \
                  steps * (self.args.epsilon_start - self.args.epsilon_min) / \
                  self.args.epsilon_decay_steps

    def get_action(self, state):
        # TODO use rng
        if random.random() <= self.epsilon:
            a = random.randint(0, self.env.num_actions-1)
        else:
            a = self.model.get_action(state)
        return a

    def step(self, iteration):
        s = self.preprocess_input(self.env.get_observation())

        eps = self.update_epsilon(iteration)

        a = self.get_action(s)
        # TODO use rng
        # if random.random() <= eps:
        #    a = random.randint(0, self.env.num_actions-1)
        # else:
        #    a = self.model.get_action(s)

        reward = self.env.step(a)
        self.rewards += reward

        is_terminal = not self.env.is_running()
        self.memory.add(s, a, reward, is_terminal)
        self.train_model()

    def train(self):
        print("Starting training.")
        train_scores = []
        self.env.reset()
        for step in range(1, self.args.steps+1):
            self.step(step)
            if not self.env.is_running():
                train_scores.append(self.rewards)
                self.rewards = 0
                self.env.reset()

            if step % self.args.backup_frequency == 0:
                self.model_name = "DQN_{:04}".format(int(step / self.args.model_frequency))
                self.model_last = os.path.join(self.model_path, self.model_name)
                print("Saving the network weights to:",
                      self.model_last,
                      file=sys.stderr)
                self.saver.save(self.session, self.model_last)  # model_name_curr)
                # making a video of the progress
                self.play()
                # TODO: Update logger to logging
                print_stats(step,
                            self.args.steps,
                            train_scores,
                            time.time() - self.start_time)

                train_scores = []

        self.env.reset()
