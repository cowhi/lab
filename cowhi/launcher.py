from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time
import sys

from cowhi.environments import LabEnvironment
from cowhi.agents import SimpleDQNAgent

__author__ = "Ruben Glatt"
__copyright__ = "Ruben Glatt"
__license__ = "MIT"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=1000,
                        help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=80,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=80,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--runfiles_path', type=str, default=None,
                        help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str, default='seekavoid_arena_01',
                        help='The environment level script to load')
    parser.add_argument('--agent', type=str, default='random_discrete',
                        help='The agent we want to use for training')
    parser.add_argument('--map', type=str, default='seekavoid_arena_01',
                         help='The map on which the agent learns.')
    parser.add_argument('--log_level', type=str, default='info',
                        help='The log level for the log file.')
    parser.add_argument('--frame_repeat', type=int, default=4,
                        help='The number of frames where an action is repeated.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='The path to a model to load for the agent.')
    return parser.parse_args()


def prepare_logging(args):
    # define and create log dir
    result_dir = "%s_%s_%s" % (
        str(time.strftime("%Y-%m-%d_%H-%M")),
        str(args.map.lower()),
        str(args.agent.lower()))
    target_dir = os.path.join(os.getcwd(), "results", result_dir)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # make sure no loggers are already active
    try:
        logging.root.handlers.pop()
    except IndexError:
        # if no logger exist the list will be empty and we
        # need to catch the resulting error
        pass

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format='[%(asctime)s][%(levelname)s][%(module)s][%(funcName)s] %(message)s',
        filename=os.path.join(target_dir, 'experiment.log'),
        filemode='w')
    _logger = logging.getLogger(__name__)
    _logger.debug("########## STARTING ##########")
    _logger.debug("parameter: %s" % str(sys.argv[1:]))
    _logger.debug("%s" % args)

    # safe experiment arguments
    args_dump = open(os.path.join(target_dir, 'args_dump.txt'), 'w', 0)
    args_dict = vars(args)
    for key in sorted(args_dict):
        args_dump.write("%s=%s\n" % (str(key), str(args_dict[key])))
    args_dump.flush()
    args_dump.close()

    return target_dir


def main(args, target_dir):
    # Initialize environment
    lab = LabEnvironment(args)
    # Initialize agent
    agent = SimpleDQNAgent(lab, args, target_dir)
    # Initialize experiment
    # experiment = SingleTaskExperiment(lab, agent, args, target_dir)
    # Train
    agent.train()


if __name__ == "__main__" and __package__ is None:
    #sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


    args = parse_args()
    target_dir = prepare_logging(args)
    # if args.runfiles_path:
    #     deepmind_lab.set_runfiles_path(args.runfiles_path)
    main(args, target_dir)
