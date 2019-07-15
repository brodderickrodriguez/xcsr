# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

import logging
from xcsr.configuration import Configuration


class ReinforcementProgram:
    def __init__(self, configuration=None):
        logging.info('ReinforcementProgram initialized')

        self.end_of_program = False
        self.time_step = 0

        try:
            self.max_steps = configuration.steps_per_episode
        except AttributeError:
            self.max_steps = Configuration().steps_per_episode

    def determine_rho(self, sigma, action):
        raise NotImplementedError

    def termination_criteria_met(self):
        raise NotImplementedError

    def reset(self):
        self.end_of_program = False
        self.time_step = 0
