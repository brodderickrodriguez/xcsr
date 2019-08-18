# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcsr.environment import Environment
import numpy as np
import logging


class Woods2Environment(Environment):
    def __init__(self, config):
        Environment.__init__(self, config)
        logging.info('WOODS2 environment initialized')

        self.state_length = 8 * 3
        self.possible_actions = [i for i in range(8)]
        self._loc_x = None
        self._loc_y = None

        self._food_types = {'F': '110', 'G': '111'}
        self._rock_types = {'O': '010', 'Q': '011'}
        self._encoding = {'.': '000'}

        self._encoding.update(self._food_types)
        self._encoding.update(self._rock_types)
        self._set_initial_state()

    @staticmethod
    def _action_map():
        return {7: (-1, -1), 0: (-1, 0), 1: (-1, +1),
                6: (0, -1), 2: (0, +1), 5: (+1, -1),
                4: (+1, 0), 3: (+1, +1)}

    def _set_initial_state(self):
        block_size = 5
        self.state = np.full((block_size * 3, block_size * 6), '.')

        for i in range(0, len(self.state), block_size):
            for j in range(0, len(self.state[i]), block_size):
                for k in range(i + 1, i + block_size - 1):
                    for l in range(j + 1, j + block_size - 1):
                        if k == i + 1 and l == j + 3:
                            choice = np.random.choice(list(self._food_types.keys()))
                        else:
                            choice = np.random.choice(list(self._rock_types.keys()))

                        self.state[k, l] = choice

        blanks = np.where(self.state == '.')
        rand_idx = int(np.floor(np.random.uniform() * len(blanks[0])))

        self._loc_y, self._loc_x = blanks[0][rand_idx], blanks[1][rand_idx]
        self.state[self._loc_y, self._loc_x] = '*'

    def step(self, action):
        self.time_step += 1
        dy, dx = self._action_map()[action]

        new_y = (self._loc_y + dy) % self.state.shape[0]
        new_x = (self._loc_x + dx) % self.state.shape[1]

        new_location = self.state[new_y, new_x]

        if new_location in self._food_types.keys():
            rho = 1000
            self.end_of_program = True
        else:
            rho = 0.01

        if new_location not in self._rock_types.keys():
            self.state[self._loc_y, self._loc_x] = '.'
            self._loc_y, self._loc_x = new_y, new_x
            self.state[self._loc_y, self._loc_x] = '*'

        return rho

    def print_world(self):
        [print(''.join(e)) for e in self.state]

    def get_state(self):
        mod_y = lambda a: a % self.state.shape[0]
        mod_x = lambda a: a % self.state.shape[1]

        vision_map = [(self._loc_y + dy, self._loc_x + dx) for dy, dx in self._action_map().values()]
        vision = [self.state[mod_y(y), mod_x(x)] for y, x in vision_map]
        raw_state = [list(self._encoding[vision[i]]) for i in range(len(vision))]

        agent_state = np.vectorize(lambda i: int(i))(raw_state).reshape(3 * 8)
        return agent_state.reshape(3 * 8)

    def reset(self):
        self.end_of_program = False
        self.time_step = 0
        self._set_initial_state()

    def termination_criteria_met(self):
        return self.time_step >= self._max_steps or self.end_of_program
