"""
Network: R -> S -> W and R -> W
"""

from bayes_class import BayesNetworkClass
from pgmpy.models import DiscreteBayesianNetwork


class WeatherClass:
    if __name__ == "__main__":
        bayes_instance = BayesNetworkClass()

        # initialize nodes and edges list
        nodes = ["R", "S", "W"]
        edges = [("R", "W"), ("R", "S"), ("S", "W")]

        # create graph
        bayesNet = DiscreteBayesianNetwork()
        bayesNet = bayes_instance.initialize_network(bayesNet, nodes, edges)

        # add CPDS for each node
        cpd_R = bayes_instance.add_cpd_to_node(
            bayesNet, "R", [[0.8], [0.2]], None, None
        )
        cpd_S = bayes_instance.add_cpd_to_node(
            bayesNet, "S", [[0.6, 0.99], [0.4, 0.01]], ["R"], [2]
        )
        cpd_W = bayes_instance.add_cpd_to_node(
            bayesNet,
            "W",
            [[1.0, 0.2, 0.1, 0.01], [0.0, 0.8, 0.9, 0.99]],
            ["R", "S"],
            [2, 2],
        )

        # check if model is correctly asdded
        bayesNet.check_model()
        print("Model is correct.")
