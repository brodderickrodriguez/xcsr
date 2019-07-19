# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcsr.environment import Environment
import numpy as np
import logging


class Woods2Environment(Environment):
    def __init__(self):
        Environment.__init__(self)
        logging.info('WOODS2 environment initialized')

        self.state_length = 8 * 3
        self.possible_actions = [i for i in range(8)]
        self.world = None
        self.loc_x = None
        self.loc_y = None
        self.food_types = {'F': '110', 'G': '111'}
        self.rock_types = {'O': '010', 'Q': '011'}
        self.encoding = {'.': '000'}
        self.encoding.update(self.food_types)
        self.encoding.update(self.rock_types)
        self.step(None)

    def _set_initial_world(self):
        block_size = 5
        self.world = np.full((block_size * 3, block_size * 6), '.')

        for i in range(0, len(self.world), block_size):
            for j in range(0, len(self.world[i]), block_size):
                for k in range(i + 1, i + block_size - 1):
                    for l in range(j + 1, j + block_size - 1):
                        if k == i + 1 and l == j + 3:
                            choice = np.random.choice(list(self.food_types.keys()))
                        else:
                            choice = np.random.choice(list(self.rock_types.keys()))

                        self.world[k, l] = choice

        blanks = np.where(self.world == '.')
        rand_idx = int(np.floor(np.random.uniform() * len(blanks[0])))

        self.loc_y, self.loc_x = blanks[0][rand_idx], blanks[1][rand_idx]

        # self.loc_y, self.loc_x = -1, -6
        self.world[self.loc_y, self.loc_x] = '*'

    def _aciton_map(self):
        return {7: (-1, -1), 0: (-1, 0), 1: (-1, +1),
                6: (0, -1), 2: (0, +1), 5: (+1, -1),
                4: (+1, 0), 3: (+1, +1)}

    def step(self, action):
        if action is None:
            self._set_initial_world()
        else:
            dy, dx = self._aciton_map()[action]

            new_y = (self.loc_y + dy) % self.world.shape[0]
            new_x = (self.loc_x + dx) % self.world.shape[1]

            new_location = self.world[new_y, new_x]

            if new_location not in self.rock_types.keys():
                self.world[self.loc_y, self.loc_x] = '.'
                self.loc_y, self.loc_x = new_y, new_x
                self.world[self.loc_y, self.loc_x] = '*'

    def print_world(self):
        [print(''.join(e)) for e in self.world]

    def get_state(self):
        y, x = self.loc_y, self.loc_x

        mod_y = lambda y: y % self.world.shape[0]
        mod_x = lambda x: x % self.world.shape[1]

        vision_map = [(y + dy, x + dx)
                      for dy, dx in self._aciton_map().values()]

        vision = [self.world[mod_y(y), mod_x(x)] for y, x in vision_map]

        raw_state = [list(self.encoding[vision[i]])
                     for i in range(len(vision))]

        self.state = np.array(raw_state).reshape(3 * 8)
        return self.state

    def reset(self):
        self.step(None)
