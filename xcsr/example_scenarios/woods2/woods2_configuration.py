# Brodderick Rodriguez
# Auburn University - CSSE
# June 29 2019

from xcsr.configuration import Configuration


class Woods2Configuration(Configuration):
    def __init__(self):
        Configuration.__init__(self)

        # the maximum number of
        self.episodes_per_repetition = 8000

        self.steps_per_episode = 100

        self.is_multi_step = True

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
