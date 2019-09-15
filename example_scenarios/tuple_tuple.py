# Brodderick Rodriguez
# Auburn University - CSSE
# 15 Sep 2019

import numpy as np
import logging
import xcsr


def build_data_set(size=100):
	states = np.floor(np.random.rand(size, 2) * 100) / 100.0
	actions = np.zeros((size, 2))

	for i, state in enumerate(states):
		actions[i, 0] = state[0] + state[1]
		actions[i, 1] = state[0] * state[1]

	return states, actions


class TupleTupleConfiguration(xcsr.Configuration):
	def __init__(self):
		xcsr.Configuration.__init__(self)

		self.state_shape = (2,)
		self.action_shape = (2,)

		# the maximum number of steps in each problem (repetition)
		self.episodes_per_repetition = 1

		self.steps_per_episode = 10 ** 4

		self.is_multi_step = False

		self.predicate_1 = 0.29

		self.predicate_delta = 0.1

		# the maximum size of the population (in micro-classifiers)
		self.N = 400

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
		self.theta_ga = 25

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
		self.p_explr = 0.01

		# the minimum number of actions that must be present in match_set
		# or else covering will occur
		# "to cause covering to provide classifiers for every action, set
		# equal to number of available actions"
		self.theta_mna = 2

		# boolean parameter. specifies if offspring are to be tested
		# for possible subsumption by parents
		self.do_ga_subsumption = True


class TupleTupleEnvironment(xcsr.Environment):
	def __init__(self, config):
		xcsr.Environment.__init__(self, config)
		logging.info('TupleTuple environment initialized')

		self._data = build_data_set(size=1000)

		self._expected_action = None

		self.possible_actions = xcsr.util.get_unique_from_ndarray(self._data[1])
		self._set_state()

	def get_state(self):
		return self._state

	def _set_state(self):
		idx = int(np.floor(np.random.uniform() * self._data[0].shape[0]))
		self._state = self._data[0][idx]
		self._expected_action = self._data[1][idx]

	def step(self, action):
		self.time_step += 1
		rho = self._determine_rho(action)
		self._set_state()
		return rho

	def _parse_action(self, action):
		_action = []
		for char in action.split(' '):
			try:
				_action.append(int(char))
			except:
				pass
		return _action

	def _determine_rho(self, action):
		self.end_of_program = True
		max_rho, rho = 1000, 0

		# if type(action) is str:
		# 	action = self._parse_action(action)

		for i in range(len(self._expected_action)):
			ai = action[i]
			eai = self._expected_action[i]

			pt_worth = float(max_rho) / float(len(action))

			if ai == eai:
				rho += pt_worth
			else:
				discounted = pt_worth * (1 / (abs(ai - eai) + 1))
				rho += discounted

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
	driver.repetitions = 5
	driver.save_location = '/Users/bcr/Desktop/ddd'
	driver.experiment_name = 'TMP'
	driver.run()

	dir_name = '{}/{}'.format(driver.save_location, driver.experiment_name)
	xcsr.util.plot_results(dir_name, title='TT', interval=50)


if __name__ == '__main__':
	# logging.basicConfig(level=logging.DEBUG)
	config = TupleTupleConfiguration
	env = TupleTupleEnvironment

	# human_play(env, config)
	run_xcsr(env, config)

