from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepmind_lab
import numpy as np
import cv2

import logging
_logger = logging.getLogger(__name__)


class Environment(object):
    def __init__(self, args):
        self.args = args


class LabEnvironment(Environment):
    def __init__(self, args):
        # Call super class
        super(LabEnvironment, self).__init__(args)
        self.env = deepmind_lab.Lab(
            self.args.level_script,
            ["RGB_INTERLACED"],
            config={
                "fps": str(self.args.fps),
                "width": str(self.args.width),
                "height": str(self.args.height)
            })
        self.reset()

        # TODO: not sure if necessary in this form
        self.action_spec = self.get_action_specs()
        self.indices = {a["name"]: i for i, a in enumerate(self.action_spec)}
        self.mins = np.array([a["min"] for a in self.action_spec])
        self.maxs = np.array([a["max"] for a in self.action_spec])
        self.num_actions = len(self.action_spec)

        # Set initial action
        self.action = None

    def reset(self):
        self.env.reset()

    def get_action_specs(self):
        return self.env.action_spec()

    def count_actions(self):
        return 3  # we only do forward and turn left/right here

    def _map_actions(self, action_raw):
        # TODO: this seems rather strange
        self.action = np.zeros([self.num_actions])
        if action_raw == 0:
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = -25
        elif action_raw == 1:
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = 25
        if action_raw == 2:  # 7
            self.action[self.indices["MOVE_BACK_FORWARD"]] = 1
        return np.clip(self.action, self.mins, self.maxs).astype(np.intc)

    def step(self, action, num_steps=None):
        if not num_steps:
            num_steps = self.args.frame_repeat
        return self.env.step(self._map_actions(action),
                             num_steps=num_steps)

    def get_observation(self):
        obs = self.env.observations()
        return cv2.cvtColor(obs["RGB_INTERLACED"], cv2.COLOR_RGB2BGR)

    def is_running(self):
        return self.env.is_running()

