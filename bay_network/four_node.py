"""
Network: A -> B, A -> C, B -> D, C -> D
"""

from bayes_class import BayesNetworkClass
from pgmpy.models import DiscreteBayesianNetwork


class FourNodeClass:
    if __name__ == "__main__":
        bayes_instance = BayesNetworkClass()

        # initialize nodes and edges list
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")]

        # create graph
        bayesNet = DiscreteBayesianNetwork()
        bayesNet = bayes_instance.initialize_network(bayesNet, nodes, edges)

        # add CPDs for each node
        cpd_A = bayes_instance.add_cpd_to_node(
            bayesNet, "A", [[0.70], [0.30]], None, None
        )
        cpd_B = bayes_instance.add_cpd_to_node(
            bayesNet, "B", [[0.8, 0.2], [0.2, 0.8]], ["A"], [2]
        )
        cpd_C = bayes_instance.add_cpd_to_node(
            bayesNet, "C", [[0.8, 0.4], [0.2, 0.6]], ["A"], [2]
        )
        cpd_D = bayes_instance.add_cpd_to_node(
            bayesNet,
            "D",
            [[0.99, 0.3, 0.1, 0.05], [0.01, 0.7, 0.9, 0.95]],
            ["B", "C"],
            [2, 2],
        )

        # check if model is correctly added
        bayesNet.check_model()
        print("Model is correct.")
