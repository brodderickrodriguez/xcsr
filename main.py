# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

import logging
import sys
import shutil
from xcsr.xcsr_driver import XCSRDriver
from xcsr import util


def human_play_rmux():
    from xcsr.example_scenarios.rmux.rmux_env import RMUXEnvironment
    from xcsr.example_scenarios.rmux.rmux_config import RMUXConfiguration

    RMUXEnvironment(config=RMUXConfiguration()).human_play()


def human_play_woods2():
    from xcsr.example_scenarios.woods2.woods2_env import Woods2Environment
    from xcsr.example_scenarios.woods2.woods2_config import Woods2Configuration

    Woods2Environment(config=Woods2Configuration()).human_play()


def rmux():
    from xcsr.example_scenarios.rmux.rmux_env import RMUXEnvironment
    from xcsr.example_scenarios.rmux.rmux_config import RMUXConfiguration

    driver = XCSRDriver()
    driver.config_class = RMUXConfiguration
    driver.env_class = RMUXEnvironment
    driver.repetitions = 5
    driver.save_location = './xcsr/example_scenarios/rmux/data'
    driver.experiment_name = 'TMP'
    driver.run()

    dir_name = driver.save_location + '/' + driver.experiment_name
    util.plot_results(dir_name, title='RMUX', interval=50)
    shutil.rmtree(dir_name)

    driver.run()


def mux():
    from xcsr.example_scenarios.mux.mux_config import MUXConfiguration
    from xcsr.example_scenarios.mux.mux_env import MUXEnvironment

    driver = XCSRDriver()
    driver.config_class = MUXConfiguration
    driver.env_class = MUXEnvironment
    driver.repetitions = 5
    driver.save_location = './xcsr/example_scenarios/mux/data'
    driver.experiment_name = 'TMP'
    driver.run()

    dir_name = driver.save_location + '/' + driver.experiment_name
    # util.plot_results(dir_name, title='MUX', interval=50)
    shutil.rmtree(dir_name)


def woods2():
    from xcsr.example_scenarios.woods2.woods2_env import Woods2Environment
    from xcsr.example_scenarios.woods2.woods2_config import Woods2Configuration

    driver = XCSRDriver()
    driver.config_class = Woods2Configuration
    driver.env_class = Woods2Environment
    driver.repetitions = 1
    driver.save_location = './xcsr/example_scenarios/woods2/data'
    driver.experiment_name = 'TMP'
    driver.run()

    dir_name = driver.save_location + '/' + driver.experiment_name
    # util.plot_results(dir_name, title='W2', interval=50)
    shutil.rmtree(dir_name)


if __name__ == '__main__':
    print('XCSR')
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # human_play_rmux()
    # rmux()
    woods2()
    # human_play_woods2()
