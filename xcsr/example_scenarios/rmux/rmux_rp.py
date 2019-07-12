# Brodderick Rodriguez
# Auburn University - CSSE
# July 12 2019

from xcsr.reinforcement_program import ReinforcementProgram

import logging
import numpy as np


class RMUXReinforcementProgram(ReinforcementProgram):
    def __init__(self, configuration=None):
        ReinforcementProgram.__init__(self, configuration)
        logging.info('RMUX rp initialized')

        self.thresholds = None

    def _get_thresholds(self, n):
        if self.thresholds is None:
            self.thresholds = [np.random.uniform() for _ in range(n)]

        return self.thresholds

    def get_sigma_bin_representation(self, sigma):
        thresholds = self._get_thresholds(len(sigma))

        bin_rep = [0 if sigma[i] < thresholds[i] else 1 for i in range(len(sigma))]
        return bin_rep

    def determine_rho(self, sigma, action):
        self.time_step += 1
        self.end_of_program = True

        if self.thresholds is None:
            self.thresholds = [np.random.uniform() for _ in range(len(sigma))]

        bin_rep = self.get_sigma_bin_representation(sigma)

        address_bits = ''.join(str(x) for x in bin_rep[:2])
        index_bit = int(address_bits, 2)
        data_bit_index = index_bit + len(address_bits)
        data_bit = bin_rep[data_bit_index]

        rho = int(data_bit == action)

        return rho

    def termination_criteria_met(self):
        return self.time_step >= self.max_steps
