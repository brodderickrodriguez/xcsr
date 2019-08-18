# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

from xcsr.environment import Environment
import numpy as np
import logging


class RMUXEnvironment(Environment):
    def __init__(self, config):
        Environment.__init__(self, config)
        logging.info('RMUX environment initialized')

        self.state_length = 6
        self.possible_actions = [0, 1]
        self._set_state()

    def get_state(self):
        return self._state

    def _set_state(self):
        self._state = [np.random.uniform() for _ in range(self.state_length)]

    def step(self, action):
        self.time_step += 1
        rho = self._determine_rho(action)
        self._set_state()
        return rho

    def _determine_rho(self, action):
        self.end_of_program = True

        address_bits = ''.join(str(round(x)) for x in self._state[:2])
        index_bit = int(address_bits, 2)
        data_bit_index = index_bit + len(address_bits)
        data_bit = round(self._state[data_bit_index])

        rho = int(data_bit == action)
        return rho

    def termination_criteria_met(self):
        return self.time_step >= self._max_steps

    def print_world(self):
        print(self._state)
