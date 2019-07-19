# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcsr.reinforcement_program import ReinforcementProgram
import numpy as np
import logging


class Woods2ReinforcementProgram(ReinforcementProgram):
    def __init__(self, configuration=None):
        ReinforcementProgram.__init__(self, configuration)
        logging.info('WOODS2 ReinforcementProgram initialized')

    @staticmethod
    def _action_map():
        return {7: (-1, -1), 0: (-1, 0), 1: (-1, +1),
                6: (0, -1), 2: (0, +1), 5: (+1, -1),
                4: (+1, 0), 3: (+1, +1)}

    def determine_rho(self, sigma, action):
        self.time_step += 1

        food_types = {'F': '110', 'G': '111'}
        encode = {'F': '110', 'G': '111', 'O': '010', 'Q': '011', '.': '000', '*': '*'}
        decode = {val: key for key, val in encode.items()}

        parsed_state = [''.join(sigma[i: i + 3]) for i in range(0, len(sigma), 3)]
        parsed_state.insert(4, '*')

        for i in range(len(parsed_state)):
            parsed_state[i] = decode[parsed_state[i]]

        vision = np.array(parsed_state).reshape((3, 3))

        dy, dx = self._action_map()[action]
        y, x = 1 + dy, 1 + dx

        if vision[y][x] in food_types.keys():
            self.end_of_program = True
            return 1000

        return -0.1

    def termination_criteria_met(self):
        return self.time_step >= self.max_steps or self.end_of_program
