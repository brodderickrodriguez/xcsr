# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

import numpy as np


class Classifier:
	PREDICATE_MIN, PREDICATE_MAX = 0.0, 0.0
	WILDCARD_ATTRIBUTE_VALUE = (PREDICATE_MIN, PREDICATE_MAX)
	CLASSIFIER_ID = 0

	def __init__(self, config, state_shape):
		self._id = Classifier.CLASSIFIER_ID
		Classifier.CLASSIFIER_ID += 1

		self._config = config

		self._state_shape = state_shape

		# condition that specifies the sensory situation which the classifier applies to
		self.predicate = [None for _ in range(self._state_shape[0])]

		# action the classifier proposes
		self.action = None

		# (p) estimated payoff expected if the classifier matches and its action is committed
		self.predicted_payoff = self._config.p_1

		# (epsilon) the error made in the predictions
		self.epsilon = self._config.epsilon_1

		# (F) the classifiers fitness
		self.fitness = self._config.F_1

		# (exp) count for the number of times this classifier has belonged to the action_set
		self.experience = 0

		# time_step of the last occurrence of a GA in an action_set to which this classifier belonged
		self.last_time_step = 0

		# (as) average size of the action_set this classifier belongs to
		self.action_set_size = 1

		# number of micro-classifiers this classifier represents
		self.numerosity = 1

	def __str__(self):
		s = '\nid: {id}\n\tpredicate: {cond}, action: {act}\n\tpred:\t{pred} \
				\n\terror:\t{err}\n\tfit:\t{fit} \
				\n\tnum:\t{num}\n\texp:\t{exp}\n\t'.format(
					id=self._id, cond=self.predicate, act=self.action,
					pred=self.predicted_payoff, err=self.epsilon,
					fit=self.fitness, num=self.numerosity, exp=self.experience)
		return s

	def __repr__(self):
		return self.__str__()

	def __lt__(self, other):
		return self.predicted_payoff < other.predicted_payoff

	def copy(self):
		other = Classifier(self._config, self._state_shape)
		other.__dict__ = self.__dict__
		return other

	def set_predicates(self, sigma):
		for i in range(self._state_shape[0]):
			if np.random.uniform() < self._config.p_sharp:
				self.predicate[i] = self.WILDCARD_ATTRIBUTE_VALUE
			else:
				h = self._config.predicate_1
				p_min = sigma[i] - np.random.uniform(high=h)
				p_max = sigma[i] + np.random.uniform(high=h)
				self.force_set_predicate_i(i, p_min, p_max)

	def force_set_predicate_i(self, i, p_min, p_max):
		p_max = min(self.PREDICATE_MAX, p_max)
		p_min = max(self.PREDICATE_MIN, p_min)
		p_min = min(p_max, p_min)
		self.predicate[i] = p_min, p_max

	def matches_sigma(self, sigma):
		items = zip(self.predicate, sigma)
		m = map(lambda x: x[0][0] <= x[1] <= x[0][1], items)

		print('sigma: {} pred: {} match: {}'.format(sigma, self.predicate, all(m)))
		return all(m)

	def does_subsume(self, other):
		# if self and other have the same action, if self is allowed to subsume and is self is more general than other
		return self.action == other.action and self.could_subsume() and self.is_more_general(other)

	def predicate_subsumes(self, other):
		for (s_p_min, s_p_max), (o_p_min, o_p_max) in zip(self.predicate, other.predicate):
			if o_p_min < s_p_min or o_p_max > s_p_max:
				return False
		return True

	def is_more_general(self, other):
		# for each attribute index i in the classifiers condition
		for i in range(self._state_shape[0]):
			# if the condition for cl_gen is not the wildcard nd cl_gen condition[i] does not match cl_spec condition[i]
			if self.predicate[i] != Classifier.WILDCARD_ATTRIBUTE_VALUE and \
					(self.predicate[0] > other.predicate[0] or self.predicate[1] < other.predicate[1]):
				# then cl_gen is not more general than cl_spec
				return False

		# otherwise, cl_gen is more general than cl_spec
		return True

	def could_subsume(self):
		# if self experience > than subsumption threshold and if self error < than the error threshold
		return self.experience > self._config.theta_sub and self.epsilon < self._config.epsilon_0
