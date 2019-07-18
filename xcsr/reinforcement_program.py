# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

import logging


class ReinforcementProgram:
    DEFAULT_MAX_STEPS = 10 ** 4

    def __init__(self, configuration=None):
        logging.info('ReinforcementProgram initialized')

        self.end_of_program = False
        self.time_step = 0

        try:
            self.max_steps = configuration.steps_per_episode
        except AttributeError:
            self.max_steps = ReinforcementProgram.DEFAULT_MAX_STEPS

    def step(self):
        self.time_step += 1

    def determine_rho(self, sigma, action):
        raise NotImplementedError()

    def termination_criteria_met(self):
        raise NotImplementedError()

    def reset(self):
        self.end_of_program = False
        self.time_step = 0
