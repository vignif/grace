import matplotlib.pyplot as plt
import matplotlib as mpl
from grace.grace import Interaction


mpl.rcParams['figure.figsize'] = (4, 6)

plt.ion()
def animate(Int: Interaction):
    plt.gca().cla() # optionally clear axes

    f1 = Int.feature_handler.available_features[0]
    f2 = Int.feature_handler.available_features[1]
    names = [f1["Feature"].name, f2["Feature"].name]
    data = [f1["Feature"].G.value, f2["Feature"].G.value]

    plt.bar(names, data)

    plt.ylim([0, 1.05])
    plt.draw()
    plt.pause(0.1)

plt.show(block=True) # block=True lets the window stay open at the end of the animation.

# for i in range(0, 20):

#     A = Agent("Human", [3, 2, 0], [0, 0, 1, 0])
#     B = Agent("Robot", [0, 2, 0], [0, 0, 0, 1])
#     # define features
#     P = ProximityFeature(A, B)
#     P.epsilon = 1
#     G = GazeFeature(A, B)

#     # feature handler
#     F = FeatureHandler(A, B)
#     F.add(P, 1.0)
#     F.add(G, 1.0)
#     F.compute()

#     Int = Interaction(F)
#     # eng = I.compute()
#     animate(Int)
