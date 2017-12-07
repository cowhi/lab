from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import numpy as np


from cowhi.environments import LabEnvironment
from cowhi.agents import SimpleDQNAgent

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

        # TODO make random number generator here
        # Mersenne Twister pseudo-random number generator
        self.rng = np.random.RandomState(self.args.random_seed)

        # Initialize environment
        self.env = LabEnvironment(self.args, self.rng)

        # Initialize agent
        self.agent = SimpleDQNAgent(self.args, self.rng, self.env, self.log_path)

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
        # Test and backup untrained agent
        self.agent.play()

        # Train the agent
        if not self.args.play:
            self.agent.train()

        # Test and backup final agent
        self.agent.play()

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
