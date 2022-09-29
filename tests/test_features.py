
from grace.grace import Interaction, Agent, FeatureHandler, ProximityFeature, GazeFeature
import numpy as np
import unittest
import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

plt_logger = logging.getLogger("grace")
plt_logger.setLevel(logging.INFO)
plt_logger.propagate = True

log = logging.getLogger(__name__)


class TestGaze(unittest.TestCase):
    def setUp(self):
        log.debug("creating common resources")

    def tearDown(self):
        log.debug("tearing down common resources")

    def get_gaze_eng(self, pose_H, pose_R):
        Human = Agent("Human", pose_H)
        Robot = Agent("Robot", pose_R)
        G = GazeFeature(Human, Robot)
        F = FeatureHandler(Human, Robot)
        F.add(G, 1.0)
        F.compute()
        I = Interaction(F)
        eng = I.compute()
        return eng

    def test_full_mutual_gaze(self):
        log.info(self.id().split('.')[-1])
        pose_H = ([3, 2, 0]), ([0, 0, 1, 0])
        pose_R = ([0, 2, 0]), ([0, 0, 0, 1])
        eng = self.get_gaze_eng(pose_H, pose_R)
        log.info(f'Engagement: {eng:.3f}')
        self.assertAlmostEqual(eng, 1.0, places=2)

    def test_only_robot_full(self):
        log.info(self.id().split('.')[-1])
        pose_H = ([3, 2, 0]), ([0, 0, 0, 1])
        pose_R = ([0, 2, 0]), ([0, 0, 0, 1])
        eng = self.get_gaze_eng(pose_H, pose_R)
        log.info(f'Engagement: {eng:.3f}')
        self.assertLess(eng, 1.0)

    def test_human_full(self):
        log.info(self.id().split('.')[-1])
        pose_H = ([3, 2, 0]), ([0, 0, 1, 0])
        pose_R = ([0, 2, 0]), ([0, 1, 0, 0])
        eng = self.get_gaze_eng(pose_H, pose_R)
        log.info(f'Engagement: {eng:.3f}')
        self.assertLess(eng, 1.0)

    def test_zero_gaze(self):
        log.info(self.id().split('.')[-1])
        pose_H = ([3, 2, 0]), ([0, 0, 0, 1])
        pose_R = ([0, 2, 0]), ([0, 0, 1, 0])
        eng = self.get_gaze_eng(pose_H, pose_R)
        log.info(f'Engagement: {eng:.3f}')
        self.assertEqual(eng, 0.0)


class TestProximity(unittest.TestCase):
    def setUp(self):
        log.debug("creating common resources")

    def tearDown(self):
        log.debug("tearing down common resources")

    def get_prox_eng(self, pose_H, pose_R):
        Human = Agent("Human", pose_H)
        Robot = Agent("Robot", pose_R)
        P = ProximityFeature(Human, Robot)
        F = FeatureHandler(Human, Robot)
        F.add(P, 1.0)
        F.compute()
        I = Interaction(F)
        eng = I.compute()
        return eng

    def test_complete(self):
        log.info(self.id().split('.')[-1])
        pose_H = ([1.5, 2, 0]), ([0, 0, 1, 0])
        pose_R = ([0, 2, 0]), ([0, 0, 0, 1])
        eng = self.get_prox_eng(pose_H, pose_R)
        log.info(f'Engagement: {eng:.3f}')
        self.assertAlmostEqual(eng, 1.0, places=2)

    def test_varying_proxemics_xy(self):
        log.info(self.id().split('.')[-1])

        pose_H = ([4, 2, 0]), ([0, 0, 1, 0])
        pose_R = ([0, 2, 0]), ([0, 0, 0, 1])

        a_pos_x = np.arange(-10.0, 10.0, 0.5)
        engs_x = []
        for x in a_pos_x:
            pose_R = ([x, 6, 0]), ([0, 0, 0, 1])
            eng = self.get_prox_eng(pose_H=pose_H, pose_R=pose_R)
            engs_x.append(eng)

        log.info(max(engs_x))

    def test_same_location(self):
        log.info(self.id().split('.')[-1])
        pose_H = ([-2, 2, 4]), ([0, 0, 1, 0])
        pose_R = ([-2, 2, 4]), ([0, 0, 0, 1])

        self.assertRaises(ValueError, self.get_prox_eng, pose_H, pose_R)
