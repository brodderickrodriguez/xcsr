# Brodderick Rodriguez
# Auburn University - CSSE
# June 6 2019

import numpy as np


class Classifier:
	WILDCARD_ATTRIBUTE_VALUE = '#'
	CLASSIFIER_ID = 0

	def __init__(self, config, state_length):
		self.id = Classifier.CLASSIFIER_ID
		Classifier.CLASSIFIER_ID += 1

		self.config = config

		self.state_length = state_length

		# condition that specifies the sensory situation
		# which the classifier applies to
		self.condition = [np.random.uniform() for _ in range(self.state_length)]

		# the interval predicate i.e. ranges which this classifier applies to
		self.interval_predicate = [np.random.uniform(0.5) for _ in range(self.state_length)]

		# action the classifier proposes
		self.action = None

		# (p) estimated payoff expected if the classifier matches and
		# its action is committed
		self.predicted_payoff = self.config.p_1

		# (epsilon) the error made in the predictions
		self.epsilon = self.config.epsilon_1

		# (F) the classifiers fitness
		self.fitness = self.config.F_1

		# (exp) count for the number of times this classifier has
		# belonged to the action_set
		self.experience = 0

		# time_step of the last occurrence of a GA in an
		# action_set to which this classifier belonged
		self.last_time_step = 0

		# (as) average size of the action_set this classifier
		# belongs to
		self.action_set_size = 0

		# number of micro-classifiers this classifier represents
		self.numerosity = 1

	def __str__(self):
		return self._to_string_detailed()

	def _to_string_simple(self):
		s = '{id} | '
		return s

	def _to_string_detailed(self):
		cond = ''.join(['{:.2f}, {:.2f}|'.format(ci, ipi) for ci, ipi in zip(self.condition, self.interval_predicate)])
		s = '{id} |{cond} act: {act} rho: {rho} err: {e} fit: {f} num: {n}'.format(id=self.id, cond=cond, act=self.action,
				rho=self.predicted_payoff, e=self.epsilon, f=self.fitness, n=self.numerosity)
		return s

	def __repr__(self):
		return self.__str__()

	def to_dict(self):
		pass

	@staticmethod
	def from_dict():
		pass

	def copy(self):
		other = Classifier(self.config, self.state_length)
		other.__dict__ = self.__dict__
		return other

	def count_wildcards(self):
		count = sum([1 if x == Classifier.WILDCARD_ATTRIBUTE_VALUE else 0 for x in self.condition])
		return count

	def matches_sigma(self, sigma):
		for ci, ipi, si in zip(self.condition, self.interval_predicate, sigma):
			if not (ci - ipi <= si < ci + ipi):
				return False
		return True

	def does_subsume(self, cl_tos):
		# if cl_sub and cl_tos have the same action
		if self.action == cl_tos.action:
			# if cl_sub is allowed to subsume another classifier
			if self.could_subsume():
				# is cl_sub is more general than cl_tos
				if self.is_more_general(cl_tos):
					# then cl_sub does subsume cl_tos
					return True

		# otherwise, cl_sub does not subsume cl_tos
		return False

	def is_more_general(self, other):
		for i in range(self.state_length):
			other_lower_bound = other.condition[i] - other.interval_predicate[i]
			other_upper_bound = other.condition[i] + other.interval_predicate[i]

			this_lower_bound = self.condition[i] - self.interval_predicate[i]
			this_upper_bound = self.condition[i] + self.interval_predicate[i]

			if other_lower_bound < this_lower_bound or other_upper_bound > this_upper_bound:
				return False

		return True

	def could_subsume(self):
		# if the classifier's experience is greater than
		# the subsumption threshold
		if self.experience > self.config.theta_sub:
			# and if the classifier's error (epsilon) is less than
			# the error threshold
			if self.epsilon < self.config.epsilon_0:
				# return true i.e. this classifier can subsume another
				return True

		# otherwise, this classifier cannot subsume another
		return False

	def deletion_vote(self, avg_fitness_in_population):
		# compute the vote-value for this classifier
		vote = self.action_set_size * self.numerosity

		# compute the weighted fitness of this classifier
		# accounting for the classifier's numerosity
		fitness_per_numerosity = self.fitness / self.numerosity

		# if this classifier's experience > the deletion threshold
		# and fitness_per_numerosity < the fraction
		# of the mean fitness in population * the average fitness
		if self.experience > self.config.theta_del and \
			fitness_per_numerosity < (self.config.delta * avg_fitness_in_population):
			# set the vote to vote * average fitness / fitness_per_numerosity
			vote = (vote * avg_fitness_in_population) / fitness_per_numerosity

		return vote
