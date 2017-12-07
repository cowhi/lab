from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time

from cowhi.experiments import Experiment
from cowhi.helper import create_dir, dump_args, prepare_logger

__author__ = "Ruben Glatt"
__copyright__ = "Ruben Glatt"
__license__ = "MIT"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Framework to guide RL in the Deepmind Lab environment")

    experiment_args = parser.add_argument_group('Experiment')
    experiment_args.add_argument('--steps', type=int, default=150,
                                 help='Number of steps to run the agent')
    experiment_args.add_argument('--log_level', type=str, default='info',
                                 help='The log level for the log file.')
    experiment_args.add_argument('--save_video', type=bool, default=False,
                                 help='If this is set a video is saved during testing.')
    experiment_args.add_argument('--show', type=bool, default=False,
                                 help='If this is set a video is shown during testing.')
    experiment_args.add_argument('--play', type=bool, default=False,
                                 help='If this is set the agent only runs some test steps.')
    experiment_args.add_argument('--backup_frequency', type=float, default=0.1,
                                 help='Frequency of model backups: backup_frequency * steps.')
    experiment_args.add_argument("--random_seed", type=int, default=666,
                                 help="Random seed for reproducible experiments.")

    environment_args = parser.add_argument_group('Environment')
    environment_args.add_argument('--width', type=int, default=80,
                                  help='Horizontal size of the observations')
    environment_args.add_argument('--height', type=int, default=80,
                                  help='Vertical size of the observations')
    environment_args.add_argument('--fps', type=int, default=60,
                                  help='Number of frames per second')
    environment_args.add_argument('--runfiles_path', type=str, default=None,
                                  help='Set the runfiles path to find DeepMind Lab data')
    environment_args.add_argument('--level_script', type=str, default='seekavoid_arena_01',
                                  help='The environment level script to load')
    environment_args.add_argument('--map', type=str, default='seekavoid_arena_01',
                                  help='The map on which the agent learns.')
    environment_args.add_argument('--color_channels', type=int, default=3,
                                  help='The number of color channels for the environment.')

    agent_args = parser.add_argument_group('Agent')
    agent_args.add_argument('--agent', type=str, default='SimpleDQNAgent',
                            help='The agent we want to use for training.')
    agent_args.add_argument('--frame_repeat', type=int, default=4,
                            help='The number of frames where an action is repeated.')
    agent_args.add_argument('--epsilon_start', type=float, default=1.0,
                            help='Exploration rate (epsilon) at the beginning of training.')
    agent_args.add_argument('--epsilon_decay_duration', type=float, default=0.66,
                            help='Percentage of total steps from starting epsilon to minimum epsilon.')
    agent_args.add_argument('--eps_min', type=float, default=0.1,
                            help='Minimum value of exploration rate (epsilon) during training.')

    model_args = parser.add_argument_group('Model')
    model_args.add_argument('--model', type=str, default='SimpleDQNModel',
                            help='The model we want to use for training.')
    model_args.add_argument('--load_model', type=str, default=None,
                            help='The path to a model to load for the agent.')
    model_args.add_argument('--alpha', type=float, default=0.00025,
                            help='The learning rate (alpha) of the model.')
    model_args.add_argument('--gamma', type=float, default=0.99,
                            help='The discount factor (gamma) of the model.')
    model_args.add_argument('--input_width', type=int, default=40,
                            help='Horizontal size of the input images for the network.')
    model_args.add_argument('--input_height', type=int, default=40,
                            help='Vertical size of the input images for the network.')
    model_args.add_argument('--batch_size', type=int, default=64,
                            help='Batch size during network training.')

    memory_args = parser.add_argument_group('Memory')
    memory_args.add_argument('--memory_size', type=int, default=1000000,
                             help='Size of the replay memory.')

    return parser.parse_args()


def main():
    # get commandline arguments
    args = parse_args()

    # define and create log path
    new_dir = "%s_%s_%s" % (
        str(time.strftime("%Y-%m-%d_%H-%M")),
        str(args.map.lower()),
        str(args.agent.lower()))
    path_to_dir = os.path.join(os.path.expanduser("~"), ".lab", new_dir)
    log_path = create_dir(path_to_dir)

    # save arguments as a text file
    dump_args(log_path, args)

    # Initialize and start logger
    prepare_logger(log_path, args.log_level)
    _logger = logging.getLogger(__name__)
    _logger.info("Start")

    # Initialize and start experiment
    experiment = Experiment(args, log_path)
    experiment.run()
    _logger.info("Finished")


if __name__ == "__main__" and __package__ is None:
    main()
