# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

from xcsr.environment import Environment
import numpy as np
import logging


class RMUXEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self)
        logging.info('RMUX environment initialized')

        self.state = None
        self.state_length = 6
        self.possible_actions = [0, 1]
        self.step(None)

    def get_state(self):
        return self.state

    def get_a(self):
        c = [float(format(i * 0.1, '2g')) for i in range(11)]
        return np.random.choice(c)

    def step(self, action):
        self.state = [np.random.uniform() for _ in range(self.state_length)]
        # self.state = [int(round(np.random.uniform())) for _ in range(self.state_length)]
        # self.state = [self.get_a() for _ in range(self.state_length)]

    def reset(self):
        self.step(None)

    def print_world(self):
        print(self.state)
