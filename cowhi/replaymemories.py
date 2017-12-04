from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np


class ReplayMemory(object):
    def __init__(self, args):
        self.args = args


class SimpleReplayMemory(ReplayMemory):
    def __init__(self, args):
        # Call super class
        super(SimpleReplayMemory, self).__init__(args)

        # TODO: put that in args
        self.channels = 3
        self.resolution = (40, 40) + (self.channels,)
        self.capacity = int(1e6)
        self.batch_size = 64

        # set the right size of the individual variables
        self.s = np.zeros((self.capacity,) + self.resolution, dtype=np.uint8)
        self.a = np.zeros(self.capacity, dtype=np.int32)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.isterminal = np.zeros(self.capacity, dtype=np.float32)

        # initialize memory
        self.size = 0
        self.pos = 0

    def add(self, s, action, isterminal, reward):
        self.s[self.pos, ...] = s
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_batch(self, sample_size):
        idx = random.sample(xrange(0, self.size - 2), sample_size)
        idx2 = []
        for i in idx:
            idx2.append(i + 1)
        return self.s[idx], self.a[idx], self.s[idx2], self.isterminal[idx], self.r[idx]
