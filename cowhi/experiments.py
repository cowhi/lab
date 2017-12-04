import logging
_logger = logging.getLogger(__name__)

import time
from collections import OrderedDict


class Experiment(object):
    """
    Base class for all experiment implementations.
    """
    def __init__(self, env, agent, args, target_dir):
        self.start_time = time.time()
        self.env = env
        self.agent = agent
        self.args = args
        self.target_dir = target_dir
        self.stats = OrderedDict()

    def reset(self):
        """ Resets all experiment data and prepares new experiment. """
        for key, _ in self.stats:
            self.stats[key] = 0.0

    def train(self):
        """ Runs a number of training steps. """
        pass

    def test(self):
        """ Runs a number of testing episodes. """
        pass

    def run(self):
        """ Organizes the training of the agent. """
        pass


class SingleTaskExperiment(Experiment):
    def __init__(self, env, agent, args, target_dir):
        # Call super class
        super(Experiment, self).__init__(env, agent, args, target_dir)
