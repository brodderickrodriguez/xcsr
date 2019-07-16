# Brodderick Rodriguez
# Auburn University - CSSE
# July 12 2019

from xcsr.classifier import Classifier

import time
import logging
import operator
import numpy as np


class XCSR:
    def __init__(self, environment, reinforcement_program, configuration):
        # all the classifiers that currently exist
        self.population = []

        # formed from population. all classifiers that their
        # condition matches the current state
        self.match_set = []

        # formed from match_set. all classifiers that propose
        # the action which was committed
        self.action_set = []

        # the action_set which was active at the previous time_step
        self.previous_action_set = []

        # the environment object
        self.env = environment

        # the reinforcement program object
        self.rp = reinforcement_program

        # the current configuration for hyper params
        self.config = configuration

        # dictionary containing all the seen rewards, expected rewards,
        # states, actions, and the number of microclassifiers
        self.metrics_history = {}
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics_history = {'rhos': [], 'predicted_rhos': [], 'microclassifier_counts': [], 'steps': 0}

    def _update_metrics(self, rho, predicted_rho):
        # save the predicted payoff from committing this action
        self.metrics_history['predicted_rhos'].append(predicted_rho)

        # save the actual payoff received for committing this action
        self.metrics_history['rhos'].append(rho)

        # save the number of microclassifiers at this time step
        num_micro_classifiers = sum([cl.numerosity for cl in self.population])
        self.metrics_history['microclassifier_counts'].append(num_micro_classifiers)

        # increment the number of steps
        self.metrics_history['steps'] += 1

    def run_experiment(self):
        np.random.seed(int(time.time()))
        previous_rho, previous_sigma = 0, []

        while not self.rp.termination_criteria_met():
            # get current situation from environment
            sigma = self.env.get_state()
            logging.debug('sigma = {}'.format(sigma))

            # generate match set. uses population and sigma
            self.match_set = self.generate_match_set(sigma)
            logging.debug('match_set = {}'.format(self.match_set))

            # generate prediction dictionary
            predictions = self.generate_prediction_dictionary()
            logging.debug('predictions = {}'.format(predictions))

            # select action using predictions
            action = self.select_action(predictions)
            logging.debug('selected action = {}'.format(action))

            # generate action set using action and match_set
            self.action_set = self.generate_action_set(action)
            logging.debug('action_set = {}'.format(self.action_set))

            # commit action
            self.env.step(action)

            # get payoff for committing this action
            rho = self.rp.determine_rho(sigma, action)
            logging.debug('payoff (rho) = {}'.format(rho))

            self._update_metrics(rho=rho, predicted_rho=predictions[action])

            # if previous_action_set is not empty
            if len(self.previous_action_set) > 0:
                # compute discounted payoff
                non_null_actions = [v for v in predictions.values() if v is not None]
                payoff = previous_rho + self.config.gamma * max(non_null_actions)

                # update previous_action_set
                self.update_set(_action_set=self.previous_action_set, payoff=payoff)

                # run genetic algorithm on previous_action_set and
                # previous_sigma inserting and possibly deleting in population
                self.run_ga(self.previous_action_set, previous_sigma)

            # if experiment is over based on information from
            # reinforcement program
            if self.rp.end_of_program:
                # update action_set
                self.update_set(_action_set=self.action_set, payoff=rho)

                # run genetic algorithm on previous_action_set and
                # previous_sigma inserting and possibly deleting in population
                self.run_ga(self.action_set, sigma)

                # empty previous_action_set
                self.previous_action_set = []
            else:
                # update previous_action_set
                self.previous_action_set = self.action_set

                # update previous rho
                previous_rho = rho

                # update previous sigma
                previous_sigma = sigma

    def generate_match_set(self, sigma):
        # local variable to hold all matching classifiers
        _match_set = []

        # continue until we have at least one classifier that matches sigma
        while len(_match_set) == 0:
            # iterate over all classifiers
            for cl in self.population:
                # check if each classifier matches the current situation (sigma)
                if cl.matches_sigma(sigma):
                    # if the classifier matches, add it to the new match set
                    _match_set.append(cl)

            # collect all the unique actions found in the local match set
            all_found_actions = set([cl.action for cl in _match_set])

            # if the length of all unique actions is less than our
            # threshold, theta_mna, begin covering procedure
            if len(all_found_actions) < self.config.theta_mna:
                # create a new classifier, cl_c
                # using the local match set and the current situation (sigma)
                cl_c = self.generate_covering_classifier(_match_set, sigma)

                # add the new classifier cl_c to the population
                self.population.append(cl_c)

                # choose individual classifiers by roulette-wheel
                # selection for deletion
                self.delete_from_population()

                # empty local match set M
                _match_set = []

        return _match_set

    def generate_covering_classifier(self, _match_set, sigma):
        # initialize new classifier
        cl = Classifier(config=self.config, state_length=self.env.state_length)

        # for each attribute in cl's condition
        for i in range(self.env.state_length):
            # if a random number is less than the probability of assigning
            # a wildcard '#'
            if np.random.uniform() < self.config.p_sharp:
                # assign it to a wildcard '#'
                cl.condition[i] = Classifier.WILDCARD_ATTRIBUTE_VALUE
            else:
                # otherwise, match the condition attribute in sigma
                cl.condition[i] = sigma[i]

        # assign a random action to this classifier that is not
        # found in the match_set
        # get all the unique actions found in the match_set
        actions_found = set([cl.action for cl in _match_set])

        # subtract the possible actions from the actions found
        difference_actions = set(self.env.possible_actions) - actions_found

        # if there are possible actions that are not in the actions_found
        if len(difference_actions) > 0:
            # find a random index in difference_actions
            rand_idx = int(np.floor(np.random.uniform() *
                                    len(difference_actions)))

            # set the action to the action corresponding to the random index
            cl.action = list(difference_actions)[rand_idx]
        else:
            # find a random index in the possible actions
            rand_idx = int(np.floor(np.random.uniform() * len(self.config.possible_actions)))

            # set the action to the action corresponding to the random index
            cl.action = self.config.possible_actions[rand_idx]

        # set the time step to the current time step
        cl.time_step = self.rp.time_step

        # set the numerosity to 1 because this method only gets called when
        # there are insufficient classifier actions
        cl.numerosity = 1

        # set the action_set_size to 1 because this method only gets called when
        # there are insufficient classifier actions
        cl.action_set_size = 1

        return cl

    def generate_prediction_dictionary(self):
        # initialize the prediction dictionary
        pa = {a: None for a in self.env.possible_actions}

        # initialize the fitness sum dictionary
        fsa = {a: 0.0 for a in self.env.possible_actions}

        # for each classifier in match_set
        for i in range(len(self.match_set)):
            # set a local variable
            cl = self.match_set[i]

            # if the value in prediction dictionary for cl.action is None
            if pa[cl.action] is None:
                # set it by accounting for fitness and predicted_payoff
                pa[cl.action] = cl.predicted_payoff * cl.fitness
            else:
                # otherwise add to the action's weighted average
                pa[cl.action] += cl.predicted_payoff * cl.fitness

            # add to the action's fitness sum
            fsa[cl.action] += cl.fitness

        # for each possible action
        for action in pa.keys():
            # if the fitness sum of the action is not zero
            if fsa[action] != 0:
                # divide by the sum of the fitness for action across
                # all classifiers
                pa[action] /= fsa[action]

        return pa

    def select_action(self, predictions):
        # select action according to an epsilon-greedy policy
        if np.random.uniform() < self.config.p_explr:
            logging.debug('selecting random action...')

            # get all actions that have some predicted value
            options = [key for key, val in predictions.items() if val is not None]

            # do pure exploration
            return np.random.choice(options)

        else:
            # get all actions that have some predicted value
            options = {key: val for key, val in predictions.items() if val is not None}

            # otherwise, return the best action to take
            best_action = max(options.items(), key=operator.itemgetter(1))

            # return the action corresponding to the highest weighted payoff
            return best_action[0]

    def generate_action_set(self, action):
        # initialize _action_set to an empty list
        _action_set = []

        # for each classifier in the match_set
        for cl in self.match_set:
            # if the classifier suggests the chosen action
            if cl.action == action:
                # add it to the action_set
                _action_set.append(cl)

        return _action_set

    # def update_set(self, _action_set, payoff):
    #     # * equations found on page 12 of 'An Algorithmic Description of XCS' *
    #     # for each classifier in A
    #     # (either self.action_set or self.previous_action_set)
    #     for cl in _action_set:
    #         # update experience
    #         cl.experience += 1
    #
    #         # if classifier experience is less than inverse of learning rate
    #         # used to determine update methods for predicted_payoff (p),
    #         # error (epsilon), and action_set_size (as)
    #         cl_exp_under_threshold = cl.experience < (1 / self.config.beta)
    #
    #         # the difference between the p-ty any predicted payoff
    #         # used to update payoff and error
    #         payoff_difference = payoff - cl.predicted_payoff
    #
    #         # the sum of the differences between each classifier's
    #         # numerosity and the action set size
    #         # used to update action set size
    #         summed_difference = np.sum([c.numerosity - cl.action_set_size for c in _action_set])
    #
    #         # update predicted_payoff (p)
    #         if cl_exp_under_threshold:
    #             cl.predicted_payoff += payoff_difference / cl.experience
    #         else:
    #             cl.predicted_payoff += self.config.beta * payoff_difference
    #
    #         # update prediction error (epsilon)
    #         if cl_exp_under_threshold:
    #             cl.epsilon += (np.abs(payoff_difference) - cl.epsilon) / cl.experience
    #         else:
    #             cl.epsilon += self.config.beta * (np.abs(payoff_difference) - cl.epsilon)
    #
    #         # update action_set_size (as)
    #         if cl_exp_under_threshold:
    #             try:
    #                 cl.action_set_size += summed_difference / cl.experience
    #             except OverflowError:
    #                 print('xcsr update as overflow', summed_difference, cl.experience)
    #         else:
    #             try:
    #                 cl.action_set_size += self.config.beta * summed_difference
    #             except OverflowError:
    #                 print('xcsr update as2 overflow', self.config.beta, summed_difference, type(summed_difference))
    #
    #     # update fitness for each classifier in A
    #     self.update_fitness(_action_set)
    #
    #     # if the program is using action_set_subsumption then
    #     # call the procedure
    #     if self.config.do_action_set_subsumption:
    #         self.do_action_set_subsumption_procedure(_action_set)

    def update_set(self, _action_set, payoff):
        for cl in _action_set:
            cl.experience += 1

            # if classifier experience is less than inverse of learning rate
            # used to determine update methods for predicted_payoff (p),
            # error (epsilon), and action_set_size (as)
            cl_exp_under_threshold = cl.experience < 1 / self.config.beta

            if cl_exp_under_threshold:
                cl.predicted_payoff += (payoff - cl.predicted_payoff) / cl.experience
            else:
                cl.predicted_payoff += self.config.beta * (payoff - cl.predicted_payoff)

            if cl_exp_under_threshold:
                cl.epsilon += (np.abs(payoff - cl.predicted_payoff) - cl.epsilon) / cl.experience
            else:
                cl.epsilon += self.config.beta * (np.abs(payoff - cl.predicted_payoff) - cl.epsilon)

            if cl_exp_under_threshold:
                cl.action_set_size += sum([c.numerosity - cl.action_set_size for c in _action_set]) / cl.experience
            else:
                cl.action_set_size += self.config.beta * sum([c.numerosity - cl.action_set_size for c in _action_set])

        self.update_fitness(_action_set)

    def update_fitness(self, _action_set):
        # set a local variable to track the accuracy over the entire set_
        accuracy_sum = 0.0

        # initialize accuracy vector (in dictionary form)
        k = {_action_set[i]: 0.0 for i in range(len(_action_set))}

        # for each classifier in set_
        for cl in _action_set:
            # if classifier error is less than the error threshold
            if cl.epsilon < self.config.epsilon_0:
                # set the accuracy to 1 (100%)
                k[cl] = 1
            else:
                k[cl] = np.power((cl.epsilon / self.config.epsilon_0), -self.config.v) * self.config.alpha

            # update accuracy_sum using a weighted sum based on
            # classifier numerosity
            accuracy_sum += k[cl] * cl.numerosity

        # for each classifier in set_
        for cl in _action_set:
            cl.fitness += self.config.beta * (((k[cl] * cl.numerosity) / accuracy_sum) - cl.fitness)

    def run_ga(self, _action_set, sigma):
        # get average time since last GA
        weighted_time = sum([cl.last_time_step * cl.numerosity for cl in _action_set])

        # get the total number of micro-classifiers currently present in set_
        num_micro_classifiers = sum([cl.numerosity for cl in _action_set])

        # compute the average time since last GA
        average_time = weighted_time / num_micro_classifiers

        # if the average time since last GA is less than the threshold
        # then do nothing
        if self.rp.time_step - average_time <= self.config.theta_ga:
            return

        # update the time since last GA for all classifiers
        for cl in _action_set:
            cl.last_time_step = self.rp.time_step

        # select two parents from the set_
        parent1 = self.select_offspring(_action_set)
        parent2 = self.select_offspring(_action_set)

        # copy each parent and create two new classifiers, child1 and child2
        child1, child2 = parent1.copy(), parent2.copy()

        # set their numerosity to 1
        child1.numerosity = child2.numerosity = 1

        # set their experience to 0
        child1.experience = child2.experience = 0

        # if a random number is less than the threshold for applying crossover
        if np.random.uniform() < self.config.chi:
            # apply crossover to child1 and child2
            self.apply_crossover(child1, child2)

            # set child1's payoff to the mean of both parents
            child1.predicted_payoff = np.mean([parent1.predicted_payoff,
                                               parent2.predicted_payoff])

            # set child1's error (epsilon) to the mean of both parents
            child1.epsilon = np.mean([parent1.epsilon, parent2.epsilon])

            # set child1's fitness to the mean of both parents
            child1.fitness = np.mean([parent1.fitness, parent2.fitness])

            # set child2's payoff to child1's payoff
            child2.predicted_payoff = child1.predicted_payoff

            # set child2's epsilon (error) to child1's epsilon (error)
            child2.epsilon = child1.epsilon

            # set child2's fitness to child1's fitness
            child2.fitness = child1.fitness

        # set child1's fitness to 10% of it parents value to verify the
        # classifier's worthiness
        child1.fitness *= 0.1

        # set child2's fitness to 10% of it parents value to verify the
        # classifier's worthiness
        child2.fitness *= 0.1

        # for both children
        for child in [child1, child2]:
            # apply mutation to child according to sigma
            self.apply_mutation(child, sigma)

            # if subsumption is true
            if self.config.do_ga_subsumption:
                # check if parent1 subsumes child
                if parent1.does_subsume(child):
                    # if it does, increment parent1 numerosity
                    parent1.numerosity += 1
                # check if parent2 subsumes child
                elif parent2.does_subsume(child):
                    # if it does, increment parent2 numerosity
                    parent2.numerosity += 1
                else:
                    # otherwise, add the child to the population of classifiers
                    self.insert_in_population(child)
            else:
                # if subsumption is false, add the child
                # to the population of classifiers
                self.insert_in_population(child)

            # choose individual classifiers by roulette-wheel
            # selection for deletion
            self.delete_from_population()

    @staticmethod
    def select_offspring(_action_set):
        # set a local variable to track the fitness over the entire set_
        fitness_sum = 0

        # for each classifier in the set_
        for cl in _action_set:
            # add its fitness to the fitness_sum
            fitness_sum += cl.fitness

        # select a random threshold for fitness_sum
        choice_point = np.random.uniform() * fitness_sum

        # reset fitness_sum to zero
        fitness_sum = 0

        # for each classifier in the set_
        for cl in _action_set:
            # add its fitness to the fitness_sum
            fitness_sum += cl.fitness

            # if we pass the choice_point, return the classifier
            # which cause us to pass the threshold
            if fitness_sum > choice_point:
                return cl

    @staticmethod
    def apply_crossover(child1, child2):
        # set a local variable for some random index in which we
        # terminate the while loop
        x = np.random.uniform() * (len(child1.condition) + 1)

        # set a local variable for some random index in which we
        # terminate the while loop
        y = np.random.uniform() * (len(child2.condition) + 1)

        # if x is greater than y
        if x > y:
            # then swap their values
            x, y = y, x

        for i in range(int(y)):
            # while we are within the random bounds specified by x and y
            if x <= i < y:
                # swap the i-th condition in child1's and child2's condition
                child1.condition[i], child2.condition[i] = child2.condition[i], child1.condition[i]

    def apply_mutation(self, child, sigma):
        # for each index in the child's condition
        for i in range(self.env.state_length):
            # if some random number is less than
            # the probability of mutating an allele in the offspring
            if np.random.uniform() < self.config.mu:
                # child.condition[i] = sigma[i]
                # if the attribute at index i is already the wildcard
                if child.condition[i] == Classifier.WILDCARD_ATTRIBUTE_VALUE:
                    # swap it with the i-th attribute in sigma
                    child.condition[i] = sigma[i]
                else:
                    # otherwise, swap it to the wildcard
                    child.condition[i] = Classifier.WILDCARD_ATTRIBUTE_VALUE

        # if some random number is less than
        # the probability of mutating an allele in the offspring
        if np.random.uniform() < self.config.mu:
            # then generate a list of all the other possible actions
            other_possible_actions = list(set(self.env.possible_actions) - {child.action})

            # find some random index in that list
            rand_idx = int(np.floor(np.random.uniform() * len(other_possible_actions)))

            # assign the action of this child to that random action
            child.action = other_possible_actions[rand_idx]

    def do_action_set_subsumption_procedure(self, set_):
        # initialize an empty classifier
        cl = None

        # create a local variable to represent the number of
        # wildcards that appear in the classifier cl's condition
        cl_wildcard_count = 0

        # for each classifier in the set_
        for c in set_:
            # if c is able to subsume other classifiers
            if c.could_subsume():
                # if cl is empty or the number of wildcards in c is greater
                # than the number of wildcards in cl or the number of
                # wildcards in c equals the number of wildcards in cl and
                # some random value is less than 0.5
                if cl is None or c.count_wildcards() > cl_wildcard_count or \
                        (cl_wildcard_count == c.count_wildcards() and np.random.uniform() < 0.5):
                    # then set cl to c
                    cl = c

                    # update the cl_wildcard_count to equal the wildcard count in c
                    cl_wildcard_count = c.count_wildcards()

        # if cl is not empty
        if cl is not None:
            # for each classifier in the set_
            for c in set_:
                # if cl is more general than c, then subsume it
                if cl.is_more_general(c):
                    # increment cl's numerosity
                    cl.numerosity += c.numerosity

                    # remove c from the set_
                    set_.remove(c)

                    # if c is in the population
                    if c in self.population:
                        # then remove it from the population
                        self.population.remove(c)

    def delete_from_population(self):
        # get the total number of micro-classifiers
        # currently present in the population
        num_micro_classifiers = sum([cl.numerosity for cl in self.population])

        # if the number of classifiers is less than the max allowed then do nothing
        if num_micro_classifiers <= self.config.N:
            return

        # the the total population for all the classifiers currently present in the population
        sum_population_fitness = sum([cl.fitness for cl in self.population])

        # compute the average fitness over all the classifiers
        # currently present in the population
        avg_fitness_in_population = sum_population_fitness / num_micro_classifiers

        # set a local variable to track the deletion vote of all the classifiers
        vote_sum = 0

        # for each classifier currently in the population
        for cl in self.population:
            # sum the deletion vote of all the classifiers
            vote_sum += cl.deletion_vote(avg_fitness_in_population)

        # select a random threshold for vote_sum
        choice_point = np.random.uniform() * vote_sum

        # reset the vote_sum to zero
        vote_sum = 0

        # for each classifier currently in the population
        for cl in self.population:
            # sum the deletion vote of all the classifiers
            vote_sum += cl.deletion_vote(avg_fitness_in_population)

            # if the current vote_sum is larger than our random threshold
            if vote_sum > choice_point:
                # if the numerosity of this classifier is > 1
                if cl.numerosity > 1:
                    # decrement its numerosity
                    cl.numerosity -= 1
                else:
                    # otherwise, if its 1, remove it from the population
                    self.population.remove(cl)

                return

    def insert_in_population(self, classifier):
        # for each classifier currently in the population
        for cl in self.population:
            # if the other classifier is equal to the parameter classifier in both condition and action
            if cl.condition == classifier.condition and cl.action == classifier.action:
                # then increment the other classifier's numerosity
                cl.numerosity += 1

                return

        # if this classifier is unique then add it to the population
        self.population.append(classifier)
