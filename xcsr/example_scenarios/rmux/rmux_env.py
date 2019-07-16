# Brodderick Rodriguez
# Auburn University - CSSE
# July 12 2019

from xcsr.environment import Environment
import numpy as np
import logging


class RMUXEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self)
        logging.info('MUX environment initialized')

        self.state = None
        self.state_length = 6
        self.possible_actions = [0, 1]
        self.step(None)

    def get_state(self):
        return self.state

    def step(self, action):
        self.state = [int(round(np.random.uniform())) for _ in range(self.state_length)]

    def reset(self):
        self.step(None)

    def print_world(self):
        print(self.state)
