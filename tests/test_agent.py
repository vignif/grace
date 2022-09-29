from grace.grace import Agent
import numpy as np
import unittest
import logging
import os
import sys

# log = Logger(__name__, logging.DEBUG).logger

os.path.join(os.path.dirname(__file__), '../grace')
sys.path.append(os.path.join(os.path.dirname(__file__), '../grace'))
# import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# from config.log_conf import Logger

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

plt_logger = logging.getLogger("matplotlib")
plt_logger.setLevel(logging.WARNING)

eng_logger = logging.getLogger("engagement")
eng_logger.setLevel(logging.WARNING)

quat_logger = logging.getLogger("numba")
quat_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)


# logger = logging.getLogger()
# logger.level = logging.DEBUG
# logger.addHandler(logging.StreamHandler(sys.stdout))

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agentA = Agent("A", [1, 2, 3], [0, 0, 0, 1])
        self.agentB = Agent("B", [1, 2, 3], [1/np.sqrt(2), 1/np.sqrt(2), 0, 0])

    def tearDown(self):
        pass
        # log.debug(f'deleting agent: {self.agentA.name}')

    def test_str_agents(self):
        log.info(self.id().split('.')[-1])
        log.info(self.agentA)
        log.info(self.agentB)

    def test_quaternion_is_not_unitary(self):
        log.info(self.id().split('.')[-1])
        self.assertRaises(AssertionError, Agent, "C", [1, 2, 3], [0, 0, 1, 1])

    def test_position_wrong(self):
        log.info(self.id().split('.')[-1])
        self.assertRaises(AssertionError, Agent, "C", [1, 2], [0, 0, 0, 1])
        self.assertRaises(AssertionError, Agent, "C",
                          [1, 2, 3, 4], [0, 0, 0, 1])

    def test_wrong_arguments(self):
        log.info(self.id().split('.')[-1])
        self.assertRaises(ValueError, Agent, "C", [1, 2, 3], [0, 0, 0, 1], 1)
        self.assertRaises(ValueError, Agent, "C", [
                          1, 2, 3], [0, 0, 0, 1], 1, 1)

    def test_mag_position(self):
        log.info(self.id().split('.')[-1])
        self.assertEqual(self.agentA.mag, np.linalg.norm(self.agentA.position))
        self.assertEqual(self.agentB.mag, np.linalg.norm(self.agentB.position))

    def test_delete_position(self):
        log.info(self.id().split('.')[-1])
        delattr(self.agentA, 'position')
        self.assertRaises(AttributeError, getattr, self.agentA, 'position')

    def test_angle_get_delete(self):
        log.info(self.id().split('.')[-1])
        self.agentA.angle = 0
        delattr(self.agentA, 'angle')
        self.assertRaises(AttributeError, getattr, self.agentA, 'angle')
