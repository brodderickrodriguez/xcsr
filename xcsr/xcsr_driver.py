# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

from xcsr.xcsr import XCSR
from xcsr.environment import Environment
from xcsr.reinforcement_program import ReinforcementProgram

import numpy as np
import multiprocessing
import logging
import os
import time
import json


class XCSRDriver:
    def __init__(self):
        logging.info('XCS Driver initialized')

        self.xcs_class = None
        self.reinforcement_program_class = None
        self.environment_class = None
        self.configuration_class = None
        self.repetitions = 10
        self.save_location = './'
        self.experiment_name = None
        self._root_data_directory = None

    def run(self):
        logging.info('Running XCSDriver')

        self._check_arguments()
        logging.info('XCSDriver passed argument check')

        self._setup_directories()
        logging.info('XCSDriver created directory: {}'.format(self._root_data_directory))

        self._run_processes()
        logging.info('XCSDriver ran all processes')

    def _check_arguments(self):
        # check if the number of repetitions is at least 1
        if self.repetitions < 1:
            raise ValueError('repetitions cannot be less than 1')

        # check if user has passed an xcs class
        if self.xcs_class is None:
            raise ValueError('xcs_class must be specified before running \
                    and it must be a class not an instance of a class')

        # check if user passed a class and not a class instance
        if isinstance(self.xcs_class, XCSR):
            raise ValueError('xcs_class cannot be an instance')

        # check if user passed an environment class
        if self.environment_class is None:
            raise ValueError('environment_class must be specified before \
                    running and it must be a class not an instance of a class')

        # check if user passed a class and not a class instance
        if isinstance(self.environment_class, Environment):
            raise ValueError('environment_class cannot be an instance')

        # check if a user passed a reinforcement_program class
        if self.reinforcement_program_class is None:
            raise ValueError('reinforcement_program_class must be specified \
                    before running and it must be a class not an instance of a class')

        # check if user passed a class and not a class instance
        if isinstance(self.reinforcement_program_class, ReinforcementProgram):
            raise ValueError('reinforcement_program_class cannot be an instance')

    def _setup_directories(self):
        time_now = str(int(time.time()))
        self.experiment_name = self.experiment_name or time_now

        self._root_data_directory = self.save_location + '/' + self.experiment_name

        directories = ['', '/classifiers', '/results', '/results/rhos', '/results/predicted_rhos',
                       '/results/microclassifier_counts', '/results/steps']

        for directory in directories:
            os.mkdir(self._root_data_directory + directory)

        metadata_file = self._root_data_directory + '/metadata.json'
        metadata = {key: val for key, val in self.configuration_class().__dict__.items()}
        metadata['repetitions'] = self.repetitions
        metadata['name'] = self.experiment_name
        metadata['root_dir'] = self._root_data_directory
        metadata['start_time'] = time_now
        f = open(metadata_file, 'w')
        json.dump(metadata, f)

    def _run_processes(self):
        if self.configuration_class().is_multi_step:
            processes = []

            for process in range(self.repetitions):
                process = multiprocessing.Process(target=self._run_multi_step_repetition, args=(process,))
                processes.append(process)
                process.start()

            print('queued all processes')

            for process in range(min(self.repetitions, len(processes))):
                processes[process].join()
                print('joined process', process)
        else:
            for process in range(self.repetitions):
                self._run_single_step_repetition(process)

    def _run_single_step_repetition(self, repetition_num):
        print('repetition {} started'.format(repetition_num))

        config = self.configuration_class()
        env = self.environment_class()
        rp = self.reinforcement_program_class(configuration=config)

        xcs_object = XCSR(environment=env, reinforcement_program=rp, configuration=config)
        xcs_object.run_experiment()

        self._save_repetition(xcs_object.metrics_history, repetition_num)

    @staticmethod
    def _post_process_episode(config, episode_metrics):
        _dict = {}

        for key, value in episode_metrics.items():
            new_val = np.array(value).astype(float)
            length = len(value) if hasattr(value, '__len__') else 1
            new_val = np.pad(new_val, [(0, config.steps_per_episode - length)], mode='constant', constant_values=np.nan)
            _dict[key] = new_val

        return _dict

    def _run_multi_step_repetition(self, repetition_num):
        print('repetition {} started'.format(repetition_num))
        config = self.configuration_class()
        env = self.environment_class()
        rp = self.reinforcement_program_class(configuration=config)
        metrics, i = [], 0

        xcs_object = XCSR(environment=env, reinforcement_program=rp, configuration=config)

        while i < config.episodes_per_repetition:
            env.reset()
            rp.reset()
            xcs_object.reset_metrics()

            config.p_explr = 0 if np.random.uniform() < 0.5 else 1

            xcs_object.run_experiment()

            if config.p_explr == 0:
                i += 1
                d = self._post_process_episode(config, xcs_object.metrics_history)
                metrics.append(d)

        metrics_compiled = {key: [] for key in metrics[0].keys()}

        for data in metrics:
            for key, value in data.items():
                metrics_compiled[key].append(value)

        self._save_repetition(metrics_compiled, repetition_num)

    def _save_repetition(self, metrics, repetition_num):
        # the path to where results are stored
        path = self._root_data_directory + '/results/'

        for key in metrics.keys():
            # the filename where we will store this metric
            filename = path + key + '/repetition' + str(repetition_num) + '.csv'

            if type(metrics[key]) is int:
                metrics[key] = [metrics[key]]

            data = np.array(metrics[key])
            np.savetxt(filename, data, delimiter=',')

        print('repetition {} done'.format(repetition_num))
