from abc import ABC, abstractmethod

# from engagement import Agent, Status
import numpy as np
import quaternion as qt
from .np_utils import angle_between_vectors
from .agent import Agent

from .sym import GaussianModel

from .mylog import Logger
log = Logger(__name__).logger


def gauss(mean):
    x = np.arange(-100, 100, 0.1)
    return np.exp(-np.square(x + mean) / 2)


def angle2shift(sign, val):
    k = 10  # constant to stretch the tan
    return sign * k * np.tan(val / 4)


class IFeature(ABC):
    def __init__(self, A: Agent, B: Agent):
        """A feature requires both agents to be calculated"""
        self.A = A
        self.B = B
        self.name = self.__class__.__name__

    @abstractmethod
    def compute():
        """Implemented in child
        assign to each agent the WtE of the feature
        """
        pass


class GazeFeature(IFeature):

    def compute(self):
        """
        compute WtE of A and B
        return both WtEs
        """

        p_a = self.A.position
        o_a = self.A.orientation

        p_b = self.B.position
        o_b = self.B.orientation

        vec_a2b = np.array(p_b - p_a)
        vec_b2a = -vec_a2b  # position of a wrt b
        log.debug(f'o_a {o_a}')
        log.debug(f'o_b {o_b}')

        B_gaze_angle = angle_between_vectors(vec_b2a, o_b.rotate([1, 0, 0]))
        A_gaze_angle = angle_between_vectors(vec_a2b, o_a.rotate([1, 0, 0]))

        log.debug(
            f"{self.A.name}'s gaze is {A_gaze_angle} far from {self.B.name}'s position")
        log.debug(
            f"{self.B.name}'s gaze is {B_gaze_angle} far from {self.A.name}'s position")

        m1 = angle2shift(-1, A_gaze_angle)
        m2 = angle2shift(1, B_gaze_angle)

        log.debug(f"m1: {m1}, m2: {m2}")
        self.G = GaussianModel(m1, m2)
        return True


class ProximityFeature(IFeature):
    epsilon = 1.5

    def compute(self):
        # print(f"compute prox")
        distance = np.linalg.norm(self.A.position - self.B.position)

        m1 = np.linalg.norm(self.A.position)
        log.debug(f"distance: {distance}")
        m2 = abs(distance - self.epsilon)
        log.debug(f"m1: {m1}, m2: {m2}")
        self.G = GaussianModel(m1, m2)
        return True


class FeatureHandler(IFeature):

    def __init__(self, A, B):
        # assign agents to base class
        super().__init__(A, B)
        self.available_features = []

    def add(self, feature, weight):
        # log.debug(weight)
        if isinstance(feature, IFeature):
            self.available_features.append({
                "Feature": feature,
                "Weight": weight
            })

    def compute(self):
        """compute and assign WtE of each feature to each agent"""
        for f in self.available_features:
            if f["Feature"].compute():
                log.info(
                    f'{f["Feature"].name} computed with value {f["Feature"].G.value}')
            else:
                log.debug(f'Error in WtE {f["Feature"].name}!')


class Interaction:

    def __init__(self, feature_handler: FeatureHandler):
        self.feature_handler = feature_handler
        self.intersections = {}
        if len(self.feature_handler.available_features) == 0:
            raise Exception("Interaction requires at least one feature")

        for feature in self.feature_handler.available_features:
            if not hasattr(feature["Feature"], 'G'):
                log.warning("Feature has no GaussianModel computed")
                raise Exception(
                    f"GaussianModel not found in feature {feature['Feature'].name} \n Did you forget to call feature.compute() ?")

    def compute(self):
        for f in self.feature_handler.available_features:
            self.intersections[f["Feature"].name] = [
                f["Feature"].G.value,
                f["Weight"],
            ]
        return self.engagement()

    def engagement(self):

        values = list(self.intersections.values())
        # log.debug(self.intersections)
        partial_eng = [row[0] for row in values]
        weights = [row[1] for row in values]
        engagement = np.average(partial_eng, axis=0, weights=weights)
        return engagement

    @staticmethod
    def get_idx(WtE_A, WtE_B):
        idx = np.argwhere(np.diff(np.sign(WtE_A - WtE_B)) != 0).reshape(-1)
        if isinstance(idx, (np.ndarray, np.generic)):
            if len(idx) == 0:
                # status_interaction[0] = Status.OVERLAP
                return np.where(WtE_A == max(WtE_A))[0][0]
            if len(idx) == 3:
                return idx[1]
            if len(idx) == 2:
                pass
                # log.error("why are there 2 intersections ??")
            if len(idx) >= 4:
                # log.warning("4 intersections found, curves overlap.")
                # status_interaction[0] = Status.OVERLAP
                return np.where(WtE_A == max(WtE_A))[0][0]

        elif isinstance(idx, float):
            # all good
            # status_interaction[0] = Status.CORRECT
            return idx
        else:
            pass
            # status_interaction[0] = Status.ERROR
        print()


def run_default(agent_A: Agent, agent_B: Agent):
    # create feature handler
    fh = FeatureHandler(agent_A, agent_B)
    # add features
    fh.add(GazeFeature(agent_A, agent_B), 1)
    fh.add(ProximityFeature(agent_A, agent_B), 1)
    # compute WtE of each feature
    fh.compute()
    # create interaction
    interaction = Interaction(fh)
    # compute engagement
    return interaction.compute()


def run(human, robot):
    # position wrt to world
    # orientation wrt to partner

    p, o = human
    q, r = robot

    A = Agent("Human", p, o)
    B = Agent("Robot", q, r)
    # define features
    P = ProximityFeature(A, B)
    P.epsilon = 1
    G = GazeFeature(A, B)

    # feature handler
    F = FeatureHandler(A, B)
    F.add(P, 1.0)
    F.add(G, 1.0)
    F.compute()

    I = Interaction(F)
    eng = I.compute()
    # log.info(f'Engagement: {eng:.3f}')
    return eng


if __name__ == "__main__":
    run(([0, 0, 0], [0, 0, 0.5, 0.5]), ([0, 1.5, 0], [0, 0, 0, 1]))


__all__ = ["Interaction", "FeatureHandler", "IFeature",
           "GazeFeature", "ProximityFeature", "Agent"]
