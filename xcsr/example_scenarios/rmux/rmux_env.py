# Brodderick Rodriguez
# Auburn University - CSSE
# July 12 2019

from xcsr.environment import Environment

import logging
import numpy as np


class RMUXEnvironment(Environment):
    def __init__(self, n=6):
        Environment.__init__(self)
        logging.info('RMUX env initialized')

        self.state_length = n
        self.possible_actions = [0, 1]
        self.step(None)

    def get_state(self):
        return self.state

    def step(self, action):
        self.state = [np.random.uniform() for _ in range(self.state_length)]

    def reset(self):
        self.step(None)

    def print_world(self):
        print(self.state)

    def human_play(self, reinforcement_program):
        while True:
            self.print_world()
            state = self.get_state()

            print(reinforcement_program.get_sigma_bin_representation(state))

            try:
                action = int(input('input action: '))
            except ValueError:
                print('invalid action')
                continue

            self.step(action)

            print('reward:\t', reinforcement_program.determine_rho(state, action))
            print('eop?:\t', reinforcement_program.end_of_program)
            print()
