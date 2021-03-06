# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

from xcsr.classifier import Classifier

import time
import logging
import numpy as np


class XCSR:
    def __init__(self, env, config):
        # all the classifiers that currently exist
        self._population = []

        # formed from population. all classifiers that their predicate matches the current state
        self._match_set = []

        # formed from match_set. all classifiers that propose the action which was committed
        self._action_set = []

        # the action_set which was active at the previous time_step
        self._previous_action_set = []

        # the environment object
        self._env = env
        # Classifier.PREDICATE_MAX = self._env.max_value
        # print('wildcard', Classifier.WILDCARD_ATTRIBUTE_VALUE)

        # the current configuration for hyper params
        self._config = config

        # dictionary containing all rewards, expected rewards, and the number of microclassifiers
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
        num_micro_classifiers = sum([cl.numerosity for cl in self._population])
        self.metrics_history['microclassifier_counts'].append(num_micro_classifiers)

        # increment the number of steps
        self.metrics_history['steps'] += 1

    def get_population(self):
        return self._population

    def run_experiment(self):
        np.random.seed(int(time.time()))
        previous_rho, previous_sigma = 0, []

        while not self._env.termination_criteria_met():
            # get current situation from environment
            sigma = self._env.get_state()
            logging.debug('sigma = {}'.format(sigma))

            # generate match set. uses population and sigma
            self._match_set = self._generate_match_set(sigma)
            logging.debug('match_set = {}'.format(self._match_set))

            # generate prediction dictionary
            predictions = self._generate_prediction_dictionary()
            logging.debug('predictions = {}'.format(predictions))

            # select action using predictions
            action = self._select_action(predictions)
            logging.debug('selected action = {}'.format(action))

            # generate action set using action and match_set
            self._action_set = self._generate_action_set(action)
            logging.debug('action_set = {}'.format(self._action_set))

            # commit action and get payoff for action
            rho = self._env.step(action)

            logging.debug('payoff (rho) = {}'.format(rho))

            self._update_metrics(rho=rho, predicted_rho=predictions[action])

            # if previous_action_set is not empty
            if len(self._previous_action_set) > 0:
                # compute discounted payoff
                non_null_actions = [v for v in predictions.values() if v is not None]
                payoff = previous_rho + self._config.gamma * max(non_null_actions)

                # update previous_action_set
                self._update_set(_action_set=self._previous_action_set, payoff=payoff)

                # run ga on previous_action_set and previous_sigma inserting and possibly deleting in population
                self._run_ga(self._previous_action_set, previous_sigma)

            # if experiment is over based on information from environment
            if self._env.end_of_program:
                # update action_set
                self._update_set(_action_set=self._action_set, payoff=rho)

                # run ga on previous_action_set and previous_sigma inserting and possibly deleting in population
                self._run_ga(self._action_set, sigma)

                # empty previous_action_set
                self._previous_action_set = []
            else:
                # update previous_action_set
                self._previous_action_set = self._action_set

                # update previous rho
                previous_rho = rho

                # update previous sigma
                previous_sigma = sigma

    def _generate_match_set(self, sigma):
        # local variable to hold all matching classifiers
        _match_set = []

        # continue until we have at least one classifier that matches sigma
        while len(_match_set) == 0:
            # add all classifiers which match sigma to _match_set
            _match_set = [cl for cl in self._population if cl.matches_sigma(sigma)]

            # collect all the unique actions found in the local match set
            all_found_actions = set([cl.action for cl in _match_set])

            # if the length of all unique actions is less than our threshold, theta_mna, begin covering procedure
            if len(all_found_actions) < self._config.theta_mna:
                # create a new classifier, cl_c using the local match set and the current situation (sigma)
                cl_c = self._generate_covering_classifier(_match_set, sigma)

                # add the new classifier cl_c to the population
                self._population.append(cl_c)

                # choose individual for deletion by beta-distributed epsilon-greedy selection
                self._delete_from_population()

                # empty local match set M
                _match_set = []

        return _match_set

    def _generate_covering_classifier(self, _match_set, sigma):
        # initialize new classifier
        cl = Classifier(config=self._config, state_shape=self._env.state_shape)

        # set the covering classifier's predicate
        cl.set_predicates(sigma)

        # get all the unique actions found in the match_set
        actions_found = set([cl.action for cl in _match_set])

        # subtract the possible actions from the actions found
        difference_actions = list(set(self._env.possible_actions) - actions_found)

        # if there are possible actions that are not in the actions_found
        if len(difference_actions) > 0:
            # find a random action in difference_actions
            choice = np.random.choice(len(difference_actions))
            cl.action = difference_actions[choice]
        else:
            # find a random action in self.config.possible_actions
            choice = np.random.choice(len(self._env.possible_actions))
            cl.action = self._env.possible_actions[choice]

        # set the time step to the current time step
        cl.time_step = self._env.time_step

        return cl

    def _generate_prediction_dictionary(self):
        # initialize the prediction dictionary
        pa = {a: None for a in self._env.possible_actions}

        # initialize the fitness sum dictionary
        fsa = {a: 0.0 for a in self._env.possible_actions}

        for cl in self._match_set:
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
                # divide by the sum of the fitness for action across all classifiers
                pa[action] /= fsa[action]

        return pa

    def _select_action(self, predictions):
        self._config.p_explr -= self._config.p_explr / self._config.steps_per_episode

        # select action according to an epsilon-greedy policy
        if np.random.uniform() < self._config.p_explr:
            logging.debug('selecting random action...')

            # get all actions that have some predicted value
            options = [key for key, val in predictions.items() if val is not None]

            # do pure exploration
            choice = np.random.choice(len(options))
            return options[choice]

        else:
            # get all actions that have some predicted value
            options = {key: val for key, val in predictions.items() if val is not None}

            # sort options by action predicted value in non-ascending order
            sorted_options = sorted(options.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

            # return the action corresponding to the highest weighted payoff
            return sorted_options[0][0]

    def _generate_action_set(self, action):
        # find all the classifiers in the match set which propose this action and return them
        return [cl for cl in self._match_set if cl.action == action]

    def _update_set(self, _action_set, payoff):
        # for each classifier in _action_set
        for cl in _action_set:
            # update experience
            cl.experience += 1

            experience_under_threshold = cl.experience < (1 / self._config.beta)

            # update predicted_payoff
            if experience_under_threshold:
                cl.predicted_payoff += (payoff - cl.predicted_payoff) / cl.experience
            else:
                cl.predicted_payoff += self._config.beta * (payoff - cl.predicted_payoff)

            # update error (epsilon)
            if experience_under_threshold:
                cl.epsilon += (np.abs(payoff - cl.predicted_payoff) - cl.epsilon) / cl.experience
            else:
                cl.epsilon += self._config.beta * (np.abs(payoff - cl.predicted_payoff) - cl.epsilon)

            summed_difference = sum([c.numerosity - cl.action_set_size for c in _action_set])

            # update action_set_size
            if experience_under_threshold:
                cl.action_set_size += summed_difference / cl.experience
            else:
                cl.action_set_size += self._config.beta * summed_difference

        # update fitness for each classifier in _action_set
        self._update_fitness(_action_set)

    def _update_fitness(self, _action_set):
        # set a local variable to track the accuracy over the entire set_
        accuracy_sum = 0.0

        # initialize accuracy vector (in dictionary form)
        k = {_action_set[i]: 0.0 for i in range(len(_action_set))}

        # for each classifier in set_
        for cl in _action_set:
            # if classifier error is less than the error threshold
            if cl.epsilon < self._config.epsilon_0:
                # set the accuracy to 100%
                k[cl] = 1
            else:
                k[cl] = ((cl.epsilon / self._config.epsilon_0) ** -self._config.v) * self._config.alpha

            # update accuracy_sum using a weighted sum based on classifier numerosity
            accuracy_sum += k[cl] * cl.numerosity

        # for each classifier in set_
        for cl in _action_set:
            cl.fitness += self._config.beta * (((k[cl] * cl.numerosity) / accuracy_sum) - cl.fitness)

    def _run_ga(self, _action_set, sigma):
        # get average time since last GA
        weighted_time = sum([cl.last_time_step * cl.numerosity for cl in _action_set])

        # get the total number of micro-classifiers currently present in set_
        num_micro_classifiers = sum([cl.numerosity for cl in _action_set])

        # compute the average time since last GA
        average_time = weighted_time / num_micro_classifiers

        # if the average time since last GA is less than the threshold then do nothing
        if self._env.time_step - average_time <= self._config.theta_ga:
            return

        # update the time since last GA for all classifiers
        for cl in _action_set:
            cl.last_time_step = self._env.time_step

        # select two parents from the set_
        parent1 = self._select_offspring(_action_set)
        parent2 = self._select_offspring(_action_set)

        # copy each parent and create two new classifiers, child1 and child2
        child1, child2 = parent1.copy(), parent2.copy()

        # set their numerosity to 1
        child1.numerosity = child2.numerosity = 1

        # set their experience to 0
        child1.experience = child2.experience = 0

        # if a random number is less than the threshold for applying crossover
        if np.random.uniform() < self._config.chi:
            # apply crossover to child1 and child2
            self._apply_crossover(child1, child2)

            # set child1 and child2 payoff to the mean of both parents
            child1.predicted_payoff = child2.predicted_payoff = \
                np.mean([parent1.predicted_payoff, parent2.predicted_payoff])

            # set child1 and child2 error (epsilon) to the mean of both parents
            child1.epsilon = child2.epsilon = np.mean([parent1.epsilon, parent2.epsilon])

            # set child1 and child2 fitness to 10% of the mean of both parents
            child1.fitness = child2.fitness = np.mean([parent1.fitness, parent2.fitness]) * 0.1

        # for both children
        for child in [child1, child2]:
            # apply mutation to child according to sigma
            self._apply_mutation(child, sigma)

            # if subsumption is true
            if self._config.do_ga_subsumption:
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
                    self._insert_in_population(child)
            else:
                # if subsumption is false, add the child to the population of classifiers
                self._insert_in_population(child)

            # choose individual for deletion by beta-distributed epsilon-greedy selection
            self._delete_from_population()

    @staticmethod
    def _select_offspring(_action_set):
        # if some random number is less than a threshold then select using a beta distribution the best classifier
        if np.random.uniform() < 0.5:
            # select a bias index
            choice_point = int(np.floor(np.random.beta(1, 5) * len(_action_set)))

            # sort the _action_set
            sorted_action_set = sorted(_action_set, reverse=True)

            # return that classifier
            return sorted_action_set[choice_point]
        else:
            # otherwise, return a random classifier
            return np.random.choice(_action_set)

    @staticmethod
    def _apply_crossover(child1, child2):
        # find two values in [0, len(predicate)) s.t. x <= y
        x = np.random.choice(range(len(child1.predicate)))
        y = np.random.choice(range(x, len(child1.predicate)))

        # swap the i-th predicate in child1's and child2's predicate
        for i in range(int(x), int(y)):
            child1.predicate[i], child2.predicate[i] = child2.predicate[i], child1.predicate[i]

    def _apply_mutation(self, child, sigma):
        # for each index in the child's predicate
        for i in range(self._env.state_shape[0]):
            # if some random number is less than the probability of mutating an allele in the offspring
            if np.random.uniform() < self._config.mu:
                # if the attribute at index i is already the wildcard
                if child.predicate[i] == Classifier.WILDCARD_ATTRIBUTE_VALUE:
                    # swap it with the i-th attribute in sigma
                    p_min = sigma[i] - self._config.predicate_1
                    p_max = sigma[i] + self._config.predicate_1
                    child.force_set_predicate_i(i, p_min, p_max)
                else:
                    # otherwise, swap it to the wildcard
                    child.predicate[i] = Classifier.WILDCARD_ATTRIBUTE_VALUE

        # if some random number is less than the probability of mutating an allele in the offspring
        if np.random.uniform() < self._config.mu:
            # then generate a list of all the other possible actions
            other_possible_actions = list(set(self._env.possible_actions) - {child.action})

            # assign the action of this child to that random action
            choice = np.random.choice(len(other_possible_actions))
            child.action = other_possible_actions[choice]

    def _delete_from_population(self):
        # if the number of classifiers is less than the max allowed the do nothing
        if self._config.N > sum([cl.numerosity for cl in self._population]):
            return

        # if some random number is less than a threshold then select using a beta distribution the best classifier
        if np.random.uniform() < 0.5:
            # select a bias index
            choice_point = int(np.floor(np.random.beta(1, 5) * len(self._population)))

            # sort the _action_set
            sorted_population = sorted(self._population, reverse=False)

            # grab the classifier corresponding to the choice point
            cl = sorted_population[choice_point]

            # if the classifier's numerosity is greater than 1 then decrement it, otherwise remove it
            if cl.numerosity > 1:
                cl.numerosity -= 1
            else:
                self._population.remove(cl)
        else:
            # otherwise choose a random classifier to delete
            cl = np.random.choice(self._population)
            self._population.remove(cl)

    def _insert_in_population(self, other):
        # for each classifier currently in the population
        for cl in self._population:
            # if the other classifier is equal to the parameter classifier in both predicate and action

            if cl.predicate_subsumes(other) and cl.action == other.action:
                # then increment the other classifier's numerosity
                cl.numerosity += 1
                return

        # if this classifier is unique then add it to the population
        self._population.append(other)
