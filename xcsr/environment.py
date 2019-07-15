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
        raise NotImplementedError
