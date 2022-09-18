import numpy as np
import unittest
import logging
from grace.grace import Interaction, Agent, FeatureHandler, ProximityFeature, GazeFeature, run
import os
import sys
os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
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


class TestInteraction(unittest.TestCase):
    def setUp(self):
        default_pose_A = ([0, 0, 0]), ([0, 0, 0, 1])
        default_pose_B = ([1, 1, 0]), ([0, 0, 0, 1])
        self.A = Agent(" A", default_pose_A)
        self.B = Agent(" B", default_pose_B)

    def tearDown(self):
        # log.debug("deleting common resources")
        pass

    def test_complete(self):
        log.info(self.id().split('.')[-1])
        P = ProximityFeature(self.A, self.B)
        P.epsilon = np.sqrt(2)
        # feature handler
        F = FeatureHandler(self.A, self.B)
        F.add(P, 1.0)
        F.compute()

        I = Interaction(F)
        eng = I.compute()
        log.info(f'Engagement: {eng:.3f}')
        # self.assertAlmostEqual(eng, 1.0, places=2)

    def test_basic_proximity(self):
        log.info(self.id().split('.')[-1])
        P = ProximityFeature(self.A, self.B)
        P.epsilon = np.sqrt(2)
        G = GazeFeature(self.A, self.B)

        # feature handler
        F = FeatureHandler(self.A, self.B)
        F.add(P, 1.0)
        F.compute()

        I = Interaction(F)
        eng = I.compute()
        log.info(f'Engagement: {eng:.3f}')
        self.assertAlmostEqual(eng, 1.0, places=2)

    def test_no_features(self):
        log.info(self.id().split('.')[-1])
        # feature handler
        F = FeatureHandler(self.A, self.B)
        self.assertRaises(Exception, Interaction, F)
        log.info('done')

    def test_only_proxemics(self):
        log.info(self.id().split('.')[-1])

        P = ProximityFeature(self.A, self.B)
        P.epsilon = np.sqrt(2)
        G = GazeFeature(self.A, self.B)

        # feature handler
        F = FeatureHandler(self.A, self.B)
        F.add(P, 1.0)
        self.assertRaises(Exception, Interaction, F)

    def test_run(self):
        run((self.A.position, self.A.orientation),
            (self.B.position, self.B.orientation))

    def test_zero_gaze(self):
        log.info(self.id().split('.')[-1])
        self.B.position = [3, 0, 0]
        G = GazeFeature(self.A, self.B)

        # feature handler
        F = FeatureHandler(self.A, self.B)
        F.add(G, 1.0)
        F.compute()
        I = Interaction(F)
        eng = I.compute()
        log.info(f'Engagement: {eng:.3f}')

    def test_weighted_full_proxemics(self):
        log.info(self.id().split('.')[-1])
        self.B.position = [1.5, 0, 0]
        # self.B.orientation = qt.as_float_array(qt.from_euler_angles(0, 0, 2*np.pi))
        P = ProximityFeature(self.A, self.B)
        P.epsilon = 1.5
        # feature handler
        F = FeatureHandler(self.A, self.B)
        F.add(P, 1.0)

        F.compute()
        I = Interaction(F)
        eng = I.compute()
        log.info(f'Engagement: {eng:.3f}')
        self.assertAlmostEqual(eng, 1.0, places=2)


class TestParameters(unittest.TestCase):
    def setUp(self):
        log.debug("creating common resources")

    def tearDown(self):
        log.debug("tearing down common resources")

    def test_prox_params(self):
        log.info(self.id().split('.')[-1])
        A = Agent(" A", ([0, 0, 0]), ([0, 0, 0, 1]))
        B = Agent(" B", ([1, 1, 0]), ([0, 0, 0, 1]))
        P = ProximityFeature(A, B)
        P.epsilon = np.sqrt(2)
        self.assertAlmostEqual(P.epsilon, 1.414, places=3)

        G = GazeFeature(A, B)
        self.assertTrue(np.array_equal(G.gaze_axis, np.array([1, 0, 0])))

#     def test_fail_on_same_location(self):
#         log.info(self.id().split('.')[-1])
#         self.A.position = np.append(self.B.position, 0)
#         self.assertRaises(ValueError, e.Interaction, self.A, self.B)
#         log.info('done')

#     def test_fail_on_angle(self):
#         log.info(self.id().split('.')[-1])
#         # robots are facing each other
#         self.A.position = np.array([10, 0, -np.pi/2])
#         self.B.position = np.array([0, 0, np.pi/2])
#         interaction = e.Interaction(self.A, self.B)
#         self.assertEqual(interaction.status, e.Status.SAME_GAZE)

#         self.A.position = np.array([-10, 2, np.pi])
#         self.B.position = np.array([0, 2, -np.pi])
#         self.assertEqual(interaction.status, e.Status.SAME_GAZE)

#         self.A.position = np.array([0, 10, 0])
#         self.B.position = np.array([0, 0, np.pi])
#         self.assertEqual(interaction.status, e.Status.SAME_GAZE)

#         self.A.position = np.array([0, 0, 0])
#         self.B.position = np.array([0, 2, np.pi])
#         self.assertEqual(interaction.status, e.Status.SAME_GAZE)
#         log.info('done')

#     def test_pass_on_angle(self):
#         log.info(self.id().split('.')[-1])
#         self.A.position = np.array([1, 0, -np.pi])
#         self.B.position = np.array([0, 0, 0.35])
#         Interaction = e.Interaction(self.A, self.B)
#         self.assertEqual(Interaction.status, e.Status.CORRECT)
#         log.info('done')

#     def test_size_engagement_intersection(self):
#         log.info(self.id().split('.')[-1])
#         Interaction = e.Interaction(self.A, self.B)
#         engagement = e.Engagement(Interaction)
#         assert isinstance(engagement.idx, np.int64)
#         log.info('done')

#     def test_B_goes_around_A(self):
#         log.info(self.id().split('.')[-1])
#         # we expect that when B will be in position (0, 2) the agents will face each other
#         self.A.position = np.array([0, 0, 0])
#         self.B.position = np.array([0, -2, np.pi])
#         Interaction = e.Interaction(self.A, self.B)
#         eng = e.Engagement(Interaction)
#         initial_proxemics_eng = eng.result([1, 0])
#         list_gaze_engagements = []
#         gazes_A = []
#         gazes_B = []
#         statuses = []
#         for angle in np.arange(0, 2*np.pi, 0.01):
#             log.debug(
#                 f"ITERATION: {angle:.2f} "
#             )
#             self.B.position = np.array(
#                 [
#                     self.B.mag * np.sin(angle),
#                     self.B.mag * np.cos(angle),
#                     self.B.angle,
#                 ]
#             )  # CHECK HERE
#             Interaction = e.Interaction(self.A, self.B)
#             eng = e.Engagement(Interaction)
#             statuses.append(Interaction.status)
#             gaze_result = eng.result([0, 1])
#             assert gaze_result <= 1.0, "Engagement can't be higher then 1.0"
#             # assert Interaction
#             gazes_A.append(eng.interaction.A.params["gaze"][1])
#             gazes_B.append(eng.interaction.B.params["gaze"][1])
#             list_gaze_engagements.append(gaze_result)
#             last_proxemics_eng = eng.result([1, 0])
#             self.assertAlmostEqual(last_proxemics_eng, initial_proxemics_eng)
#         plt.plot(list_gaze_engagements)
#         plt.show()

#         print()
#         log.info('done')

#     def test_B_goes_around_A_in_neighbor(self):
#         log.info(self.id().split('.')[-1])
#         # we expect that when B will be in position (0, 2) the agents will face each other
#         self.A.position = np.array([0, 0, 0])
#         self.B.position = np.array([0, 4, np.pi])
#         Interaction = e.Interaction(self.A, self.B)
#         eng = e.Engagement(Interaction)
#         initial_proxemics_eng = eng.result([1, 0])
#         list_gaze_engagements = []
#         gazes_A = []
#         gazes_B = []
#         statuses = []
#         for angle in np.arange(5/6 * np.pi, 7/6*np.pi, 0.0001):
#             log.debug(
#                 f"ITERATION: {angle:.2f} "
#             )
#             self.B.position = np.array(
#                 [
#                     self.B.mag * np.sin(angle),
#                     self.B.mag * np.cos(angle),
#                     self.B.angle,
#                 ]
#             )  # CHECK HERE
#             Interaction = e.Interaction(self.A, self.B)
#             eng = e.Engagement(Interaction)
#             log.info(f'{angle}')
#             statuses.append(Interaction.status)
#             gaze_result = eng.result([0, 1])
#             assert gaze_result <= 1.0, "Engagement can't be higher then 1.0"
#             # assert Interaction
#             gazes_A.append(eng.interaction.A.params["gaze"][1])
#             gazes_B.append(eng.interaction.B.params["gaze"][1])
#             list_gaze_engagements.append(gaze_result)
#             last_proxemics_eng = eng.result([1, 0])
#             self.assertAlmostEqual(last_proxemics_eng, initial_proxemics_eng)
#         plt.plot(list_gaze_engagements)
#         plt.show()
#         # check that at least one interaction contains the state "SAME_GAZE"
#         stats_with_same_gaze = [s for s in statuses if s.value == 0 ]
#         self.assertGreaterEqual(len(stats_with_same_gaze), 1)
#         print()
#         log.info('done')


if __name__ == "__main__":
    unittest.main()
