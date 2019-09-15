# Brodderick Rodriguez
# Auburn University - CSSE
# 15 Sep 2019

import numpy as np
import logging
import shutil
import xcsr


class Woods2Environment(xcsr.Environment):
    def __init__(self, config):
        xcsr.Environment.__init__(self, config)
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
                    for m in range(j + 1, j + block_size - 1):
                        if k == i + 1 and m == j + 3:
                            choice = np.random.choice(list(self._food_types.keys()))
                        else:
                            choice = np.random.choice(list(self._rock_types.keys()))

                        self.state[k, m] = choice

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
        vision_map = [(self._loc_y + dy, self._loc_x + dx) for dy, dx in self._action_map().values()]
        vision_map = [(y % self.state.shape[0], x % self.state.shape[1]) for y, x in vision_map]

        vision = [self.state[y, x] for y, x in vision_map]
        raw_state = [list(self._encoding[vision[i]]) for i in range(len(vision))]

        agent_state = np.vectorize(lambda i: int(i))(raw_state).reshape(3 * 8)
        return agent_state.reshape(3 * 8)

    def reset(self):
        self.end_of_program = False
        self.time_step = 0
        self._set_initial_state()

    def termination_criteria_met(self):
        return self.time_step >= self._max_steps or self.end_of_program


class Woods2Configuration(xcsr.Configuration):
    def __init__(self):
        xcsr.Configuration.__init__(self)

        # the maximum number of steps in each problem (repetition)
        self.episodes_per_repetition = 8000

        self.steps_per_episode = 100

        self.is_multi_step = True

        self.predicate_1 = 0.29

        self.predicate_delta = 0.1

        # the maximum size of the population (in micro-classifiers)
        self.N = 800

        self.mu = 0.01
        self.p_sharp = 0.5

        # learning rate for payoff, epsilon, fitness, and action_set_size
        self.beta = 0.1

        # used to calculate the fitness of a classifier
        self.alpha = 0.1
        self.epsilon_0 = 0.01
        self.v = 5

        # discount factor
        self.gamma = 0.9

        self.chi = 0.8

        self.delta = 0.1

        # the GA threshold. GA is applied in a set when the average time
        # since the last GA in the set is greater than theta_ga
        self.theta_ga = 25

        # used as initial values in new classifiers
        self.p_1 = 10
        self.epsilon_1 = 0
        self.F_1 = 10

        # probability during action selection of choosing the
        # action uniform randomly
        self.p_explr = 0.5

        self.theta_mna = 8

        # boolean parameter. specifies if offspring are to be tested
        # for possible subsumption by parents
        self.do_ga_subsumption = True

        # boolean parameter. specifies if action sets are to be tested
        # for subsuming classifiers
        self.do_action_set_subsumption = False


def human_play(ENV, CONFIG):
    ENV(config=CONFIG()).human_play()



def run_xcsr(ENV, CONFIG):
    driver = xcsr.XCSRDriver()
    driver.config_class = CONFIG
    driver.env_class = ENV
    driver.repetitions = 5
    driver.save_location = '/Users/bcr/Desktop/ddd'
    driver.experiment_name = 'TMP'
    driver.run()

    dir_name = '{}/{}'.format(driver.save_location, driver.experiment_name)
    xcsr.util.plot_results(dir_name, title='W2', interval=50)
    shutil.rmtree(dir_name)


if __name__ == '__main__':
	config = Woods2Configuration
	env = Woods2Environment

	# human_play(env, config)
	run_xcsr(env, config)
