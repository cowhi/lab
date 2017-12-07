from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import logging
import os
import time
import numpy as np
import tensorflow as tf

from cowhi.environments import LabEnvironment
from cowhi.agents import SimpleDQNAgent
from cowhi.models import SimpleDQNModel
from cowhi.replaymemories import SimpleReplayMemory
from cowhi.helper import create_dir

_logger = logging.getLogger(__name__)


class Experiment(object):
    """
    Base class for all experiment implementations.
    """
    def __init__(self, args, log_path):
        _logger.info("Initializing Experiment (Steps: %i)" %
                     args.steps)
        self.args = args
        self.log_path = log_path

        # Initialize important stats
        self.time_start = time.time()
        self.time_current = time.time()
        self.episodes = 0
        self.episodes_success = 0
        self.steps_current = 0
        self.steps_episode = 0
        self.reward_total = 0
        self.reward_episode = 0

        # TODO Initialize important environment variables
        self.model_input_shape = (self.args.input_width, self.args.input_height) + \
                                 (self.args.color_channels,)
        self.model_path = create_dir(os.path.join(log_path, 'models'))
        self.video_path = create_dir(os.path.join(log_path, 'videos'))

        # Mersenne Twister pseudo-random number generator
        self.rng = np.random.RandomState(self.args.random_seed)

        # Initialize environment
        self.env = LabEnvironment(self.args, self.rng)

        # TODO Initialize replay memory
        self.memory = SimpleReplayMemory(self.args,
                                         self.rng,
                                         self.model_input_shape)

        # Initialize agent
        self.agent = SimpleDQNAgent(self.args, self.rng, self.env, self.log_path)

        # TODO Prepare model
        tf.set_random_seed(self.args.random_seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True
        # Initiate tensorflow session
        self.session = tf.Session(config=config)

        # TODO Initialize model
        self.model = SimpleDQNModel(self.args,
                                    self.rng,
                                    self.session,
                                    self.model_input_shape,
                                    self.env.num_actions)

        # TODO set model parameters
        self.saver = tf.train.Saver(max_to_keep=1000)
        if self.args.load_model is not None:
            self.saver.restore(self.session, self.args.load_model)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)
        self.model_name = "DQN_0000"
        self.model_last = os.path.join(self.model_path, self.model_name)
        self.saver.save(self.session, self.model_last)

    def preprocess_input(self, img):
        if self.args.color_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.args.input_width, self.args.input_height))
        return np.reshape(img,
                          self.model_input_shape)

    def reset(self):
        """ Resets all experiment data and prepares new experiment. """
        pass

    def train(self):
        """ Runs a number of training steps. """
        pass

    def test(self):
        """ Runs a number of testing episodes. """
        pass

    def run(self):
        """ Organizes the training of the agent. """

        if not self.args.play:
            # Test and backup untrained agent
            self.agent.play()
            # Train the agent
            self.agent.train()

        # Test and backup final agent
        self.agent.play()

        print("Starting training.")
        train_scores = []
        self.env.reset()
        for step in range(1, self.args.steps + 1):
            # TODO Perform agent step
            # self.agent.step(step)
            self.agent.epsilon = self.agent.update_epsilon(step)
            s = self.preprocess_input(self.env.get_observation())
            a = self.agent.get_action()
            r = self.env.step(a)
            is_terminal = not self.env.is_running()
            # TODO add feedback to memory
            self.memory.add(s, a, r, is_terminal)
            # TODO Update stats
            self.reward_episode += r
            # TODO Perform training step on model
            self.train_model()

            if not self.env.is_running():
                train_scores.append(self.agent.rewards)
                self.agent.rewards = 0  # TODO put in episode reset
                self.env.reset()  # TODO put in episode reset

            if step % self.args.backup_frequency == 0:
                # TODO put in init with model definitions
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


        """ As it should be 

        train_scores = []
        self.env.reset()
        for step in xrange(1, self.args.steps + 1):
            state = self.preprocess_observation(self.env.get_observation())
            self.agent.step(step)
            if self.env.episode_ended():
                train_scores.append(self.rewards)
                self.rewards = 0
                self.env.reset()

            if step % self.model_backup_frequency == 0:
                self.model_name = "DQN_{:04}".format(int(step / self.model_backup_frequency))
                self.model_last = os.path.join(self.model_path, self.model_name)
                print("\nSaving the network weights to:",
                      self.model_last,  # model_name_curr,
                      file=sys.stderr)
                self.saver.save(self.session, self.model_last)  # model_name_curr)
                # making a video of the progress
                self.play()
                # TODO: Update logger to logging
                print_stats(step,
                            self.args.steps,
                            train_scores,
                            time.time() - self.time_start)

                train_scores = []

        self.env.reset()
        
        """
