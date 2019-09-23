# Brodderick Rodriguez
# Auburn University - CSSE
# 15 Sep 2019

import numpy as np
import logging
import shutil
import xcsr


class RMUXConfiguration(xcsr.Configuration):
    def __init__(self):
        xcsr.Configuration.__init__(self)

        # the maximum number of steps in each problem (replication)
        self.episodes_per_replication = 1

        self.steps_per_episode = 10 ** 4

        self.is_multi_step = False

        self.predicate_1 = 0.29

        self.predicate_delta = 0.1

        # the maximum size of the population (in micro-classifiers)
        self.N = 10000

        # learning rate for payoff, epsilon, fitness, and action_set_size
        self.beta = 0.2

        # used to calculate the fitness of a classifier
        self.alpha = 0.1
        self.epsilon_0 = 0.01
        self.v = 5

        # discount factor
        self.gamma = 0.71

        # the GA threshold. GA is applied in a set when the average time
        # since the last GA in the set is greater than theta_ga
        self.theta_ga = 12

        # the probability of applying crossover in the GA
        self.chi = 0.8

        # specifies the probability of mutating an allele in the offspring
        self.mu = 0.04

        # subsumption threshold. experience of a classifier must be greater
        # than theta_sub in order to be able to subsume another classifier
        self.theta_sub = 20

        # probability of using '#' (Classifier.WILDCARD_ATTRIBUTE_VALUE)
        # in one attribute in the condition of a classifier when covering
        self.p_sharp = 0.33

        # used as initial values in new classifiers
        self.p_1 = 0.01
        self.epsilon_1 = 0.0
        self.F_1 = 0.01

        # probability during action selection of choosing the
        # action uniform randomly
        self.p_explr = 0.5

        # the minimum number of actions that must be present in match_set or else covering will occur
        # "to cause covering to provide classifiers for every action, set equal to number of available actions"
        self.theta_mna = 2

        # boolean parameter. specifies if offspring are to be tested
        # for possible subsumption by parents
        self.do_ga_subsumption = True


class RMUXEnvironment(xcsr.Environment):
    def __init__(self, config):
        xcsr.Environment.__init__(self, config)
        logging.info('RMUX environment initialized')

        self.state_shape = (6,)
        self.action_shape = (1,)
        self.possible_actions = [(0,), (1,)]
        self._set_state()

    def get_state(self):
        return self._state

    def _set_state(self):
        self._state = [np.random.uniform() for _ in range(self.state_shape[0])]

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

        rho = int(data_bit == action[0])
        return rho

    def termination_criteria_met(self):
        return self.time_step >= self._max_steps

    def print_world(self):
        print(self._state)


def human_play(ENV, CONFIG):
    ENV(config=CONFIG()).human_play()


def run_xcsr(ENV, CONFIG):
    driver = xcsr.XCSRDriver()
    driver.config_class = CONFIG
    driver.env_class = ENV
    driver.replications = 5
    driver.save_location = '/Users/bcr/Desktop/'
    driver.experiment_name = 'TMP'
    driver.run()

    dir_name = '{}/{}'.format(driver.save_location, driver.experiment_name)
    xcsr.util.plot_results(dir_name, title='RMUX', interval=50)
    shutil.rmtree(dir_name)


if __name__ == '__main__':
    config = RMUXConfiguration
    env = RMUXEnvironment

    # human_play(env, config)
    run_xcsr(env, config)
