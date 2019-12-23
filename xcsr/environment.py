# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

import logging


class Environment:
    def __init__(self, config, *args):
        logging.info('environment initialized')
        self.state_shape = (0,)
        self.action_shape = (1,)
        self.possible_actions = [(None,)]
        self.end_of_program = False
        self.time_step = 0
        self.max_value = 1.0

        self._max_steps = config.steps_per_episode
        self._state = None

    def get_state(self):
        raise NotImplementedError()

    def termination_criteria_met(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        self.end_of_program = False
        self.time_step = 0

    def print_world(self):
        raise NotImplementedError()

    def human_play(self):
        while not self.termination_criteria_met():
            self.print_world()

            print(self.get_state())

            try:
                action = input('input action: ')
            except ValueError:
                print('invalid action')
                continue

            rho = self.step(action)

            print('reward:\t', rho)
            print('eop?:\t', self.end_of_program)
            print()
