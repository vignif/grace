"""
GRACE GeometRic ApproaCh to mutual Engagement

Author: Francesco Vigni
"""

import numpy as np
from grace.utils import Agent, Logger
from grace.features import GazeFeature, IFeature, ProximityFeature, FeatureHandler
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.figsize'] = (4, 6)


log = Logger(__name__).logger


class Grace:
    def __init__(self, agents: list, features: list):
        try:
            assert all(isinstance(agent, Agent) for agent in agents)
        except AssertionError:
            raise AssertionError("agents must be a list of Agent objects")

        try:
            for feature in features:
                assert feature is not None, "features must be a list of IFeature objects"
                assert issubclass(feature, IFeature)
        except AssertionError:
            raise AssertionError("All features must be of type IFeature")

        if len(agents) != 2:
            raise ValueError("Only two agents are supported")

        if id(agents[0]) == id(agents[1]):
            raise ValueError("Agents must be different")

        self.mutual_engagement = None
        self.agents = agents
        self.fh = FeatureHandler(self.agents[0], self.agents[1])
        for feature in features:
            instance_feature = feature(self.agents[0], self.agents[1])
            self.fh.add(instance_feature, 1.0)
        self.fh.compute()

    def update(self, feature, weight):
        self.fh.update(feature, weight)

    def compute(self):
        """Compute the mutual engagement of the two agents
        """
        self.interaction = Interaction(self.fh)
        self.mutual_engagement = self.interaction.compute()
        return self.mutual_engagement

    def visualize_features(self):
        animate(self.interaction)


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


def animate(interaction):
    # TODO still not stable
    mpl.rcParams['figure.figsize'] = (4, 6)

    plt.ion()
    def run_animation(interaction: Interaction):
        plt.gca().cla() # optionally clear axes

        f1 = interaction.feature_handler.available_features[0]
        f2 = interaction.feature_handler.available_features[1]
        names = [f1["Feature"].name, f2["Feature"].name]
        data = [f1["Feature"].G.value, f2["Feature"].G.value]

        plt.bar(names, data)

        plt.ylim([0, 1.05])
        plt.draw()
        plt.pause(0.1)
    run_animation(interaction)
    plt.show(block=True) # block=True lets the window stay open at the end of the animation.

if __name__ == "__main__":
    run(([0, 0, 0], [0, 0, 0, 1]), ([1, 0, 0], [0, 0, 0, 1]))


__all__ = ["Interaction", "Grace"]
