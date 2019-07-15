# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

import logging


class Environment:
    def __init__(self):
        logging.info('environment initialized')

        self.state = None
        self.state_length = 0
        self.possible_actions = []

    def get_state(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def print_world(self):
        raise NotImplementedError

    def human_play(self, reinforcement_program):
        while not reinforcement_program.termination_criteria_met():
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
