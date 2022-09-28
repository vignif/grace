

from abc import ABC, abstractmethod
from grace.utils import Agent, Logger, angle_between_vectors, angle2shift, GaussianModel
import numpy as np

log = Logger(__name__).logger


class IFeature(ABC):
    def __init__(self, A: Agent, B: Agent):
        """A feature requires both agents to be calculated"""
        self.A = A
        self.B = B
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self):
        """Implemented in child
        assign to each agent the WtE of the feature
        """


class GazeFeature(IFeature):
    # TODO: add a parameter to set the angle of the cone

    # gaze_axis is the x component of the pose
    gaze_axis = np.array([1, 0, 0])

    def compute(self):
        """Compute the gaze feature
        """

        p_a = self.A.position
        o_a = self.A.orientation

        p_b = self.B.position
        o_b = self.B.orientation

        vec_a2b = np.array(p_b - p_a)
        vec_b2a = -vec_a2b  # position of a wrt b
        log.debug(f'o_a {o_a}')
        log.debug(f'o_b {o_b}')

        B_gaze_angle = angle_between_vectors(
            vec_b2a, o_b.rotate(self.gaze_axis))
        A_gaze_angle = angle_between_vectors(
            vec_a2b, o_a.rotate(self.gaze_axis))

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
    # TODO: epsilon as function of time

    # epsilon is ideal interaction distance
    epsilon = 1.5

    def __init__(self, A: Agent, B: Agent):
        super().__init__(A, B)
        if np.array_equal(self.A.position, self.B.position):
            raise ValueError(
                f"{self.A.name} and {self.B.name} are on the same location")

    def compute(self):
        """Compute the proximity feature
        """
        m1 = np.linalg.norm(self.A.position - self.B.position) - self.epsilon
        m2 = 0.0

        log.debug(f"m1: {m1}, m2: {m2}")
        self.G = GaussianModel(m1, m2)
        return True


class FeatureHandler(IFeature):

    def __init__(self, A, B):
        """Assign agents to base class
        """
        super().__init__(A, B)
        self.available_features = []

    def add(self, feature, weight):
        """add a feature to the list of available features"""
        if isinstance(feature, IFeature):
            self.available_features.append({
                "Feature": feature,
                "Weight": weight
            })

    def update(self, feature, weight):
        """update the weight of a feature"""
        for f in self.available_features:
            if isinstance(f["Feature"], feature):
                f["Weight"] = weight
                log.debug(f'Weight of {f["Feature"].name} updated to {weight}')

    def compute(self):
        """compute and assign WtE of each feature to each agent"""
        for f in self.available_features:
            if f["Feature"].compute():
                log.info(
                    f'{f["Feature"].name} computed with value {f["Feature"].G.value}')
            else:
                log.debug(f'Error in WtE {f["Feature"].name}!')
