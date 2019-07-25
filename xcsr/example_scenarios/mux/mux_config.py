# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

from xcsr.configuration import Configuration


class MUXConfiguration(Configuration):
    def __init__(self):
        Configuration.__init__(self)

        # the maximum number of
        self.episodes_per_repetition = 1

        self.steps_per_episode = 1.5 * 10 ** 4

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
