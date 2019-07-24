# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

import numpy as np


class Classifier:
	WILDCARD_ATTRIBUTE_VALUE = '#'
	CLASSIFIER_ID = 0

	def __init__(self, config, state_length):
		self.id = Classifier.CLASSIFIER_ID
		Classifier.CLASSIFIER_ID += 1

		self.config = config

		self.state_length = state_length

		# condition that specifies the sensory situation which the classifier applies to
		self.predicate = [0 for _ in range(self.state_length)]

		# action the classifier proposes
		self.action = None

		# (p) estimated payoff expected if the classifier matches and its action is committed
		self.predicted_payoff = self.config.p_1

		# (epsilon) the error made in the predictions
		self.epsilon = self.config.epsilon_1

		# (F) the classifiers fitness
		self.fitness = self.config.F_1

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
					id=self.id, cond=self.predicate, act=self.action,
					pred=self.predicted_payoff, err=self.epsilon,
					fit=self.fitness, num=self.numerosity, exp=self.experience)
		return s

	def __repr__(self):
		return self.__str__()

	def __lt__(self, other):
		return self.predicted_payoff < other.predicted_payoff

	def copy(self):
		other = Classifier(self.config, self.state_length)
		other.__dict__ = self.__dict__
		return other

	def set_predicates(self, sigma):
		# for each attribute in cl's condition
		for i in range(self.state_length):
			# if a random number is less than the probability of assigning a wildcard '#'
			if np.random.uniform() < self.config.p_sharp:
				# assign it to a wildcard '#'
				self.predicate[i] = Classifier.WILDCARD_ATTRIBUTE_VALUE
			else:
				# otherwise, match the condition attribute in sigma
				self.predicate[i] = sigma[i]

	def count_wildcards(self):
		count = sum([1 if x == Classifier.WILDCARD_ATTRIBUTE_VALUE else 0 for x in self.predicate])
		return count

	def matches_sigma(self, sigma):
		for ci, si in zip(self.predicate, sigma):
			if ci != Classifier.WILDCARD_ATTRIBUTE_VALUE and ci != si:
				return False
		return True

	def does_subsume(self, other):
		# if self and other have the same action, if self is allowed to subsume and is self is more general than other
		return self.action == other.action and self.could_subsume() and self.is_more_general(other)

	def predicate_is_more_general(self, other):
		pass

	def is_more_general(self, other):
		# count the number of wildcards in cl_gen
		wildcard_count = self.count_wildcards()

		# count the number of wildcards in cl_spec
		other_wildcard_count = other.count_wildcards()

		# if cl_gen is not more general than cl_spec
		if wildcard_count <= other_wildcard_count:
			return False

		# for each attribute index i in the classifiers condition
		for i in range(self.state_length):
			# if the condition for cl_gen is not the wildcard nd cl_gen condition[i] does not match cl_spec condition[i]
			if self.predicate[i] != Classifier.WILDCARD_ATTRIBUTE_VALUE and self.predicate[i] != other.predicate[i]:
				# then cl_gen is not more general than cl_spec
				return False

		# otherwise, cl_gen is more general than cl_spec
		return True

	def could_subsume(self):
		# if self experience > than subsumption threshold and if self error < than the error threshold
		return self.experience > self.config.theta_sub and self.epsilon < self.config.epsilon_0
