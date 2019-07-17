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
        self.state = [round(np.random.uniform() * 1000 / 1000, 3) for _ in range(self.state_length)]

    def reset(self):
        self.step(None)

    def print_world(self):
        print(self.state)

    def human_play(self, reinforcement_program):
        print('rmux env')

        while True:  # not reinforcement_program.termination_criteria_met():
            self.print_world()

            state = self.get_state()

            try:
                action = int(input('input action: '))
            except ValueError:
                print('invalid action')
                continue

            self.step(action)

            print('reward:\t', reinforcement_program.determine_rho(state, action))
            print('eop?:\t', reinforcement_program.end_of_program)
            print()
