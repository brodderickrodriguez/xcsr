# Brodderick Rodriguez
# Auburn University - CSSE
# july 12 2019

import logging
import sys
import shutil
from xcsr.xcsr_driver import XCSRDriver
from xcsr.xcsr import XCSR
from xcsr import util


def human_play_rmux():
    from xcsr.example_scenarios.rmux.rmux_env import RMUXEnvironment
    from xcsr.example_scenarios.rmux.rmux_rp import RMUXReinforcementProgram

    rp = RMUXReinforcementProgram()
    env = RMUXEnvironment()
    env.human_play(reinforcement_program=rp)


def rmux():
    from xcsr.example_scenarios.rmux.rmux_env import RMUXEnvironment
    from xcsr.example_scenarios.rmux.rmux_rp import RMUXReinforcementProgram
    from xcsr.example_scenarios.rmux.rmux_config import RMUXConfiguration

    driver = XCSRDriver()
    driver.repetitions = 3
    driver.save_location = './xcsr/example_scenarios/rmux/data'
    driver.save_location = '/home/bcr0012/xcsr/xcsr/example_scenarios/rmux/data'
    driver.experiment_name = 'good'

    driver.xcs_class = XCSR
    driver.environment_class = RMUXEnvironment
    driver.reinforcement_program_class = RMUXReinforcementProgram
    driver.configuration_class = RMUXConfiguration

    driver.run()

    # dir_name = './xcsr/example_scenarios/rmux/data/' + driver.experiment_name
    # util.plot_results(dir_name, title='RMUX', interval=100)
    # shutil.rmtree('./xcsr/example_scenarios/rmux/data/TMP')


def woods2():
    from xcsr.example_scenarios.woods2.woods2_environment import Woods2Environment
    from xcsr.example_scenarios.woods2.woods2_reinforcement_program import Woods2ReinforcementProgram
    from xcsr.example_scenarios.woods2.woods2_configuration import Woods2Configuration

    driver = XCSRDriver()
    driver.repetitions = 5
    driver.save_location = './xcsr/example_scenarios/woods2/data'
    driver.experiment_name = 'TMP'

    driver.xcs_class = XCSR
    driver.environment_class = Woods2Environment
    driver.reinforcement_program_class = Woods2ReinforcementProgram
    driver.configuration_class = Woods2Configuration

    driver.run()

    dir_name = './xcsr/example_scenarios/woods2/data/' + driver.experiment_name
    util.plot_results(dir_name, title='W2', interval=50)

    shutil.rmtree('./xcsr/example_scenarios/woods2/data/TMP')


if __name__ == '__main__':
    print('XCSR')
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # human_play_rmux()
    rmux()
    # woods2()
