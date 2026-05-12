"""
The variable elimination algorithm for exact inference in Bayesian networks.
"""

from bayes_class import BayesNetworkClass
from pgmpy.models import DiscreteBayesianNetwork


def elimination_ask(X, e, bn):
    """
    Returns the probability distribution of the query variable X given the observed values e in the Bayesian network bn with variables vars.
    """
    factors = []
    for var in order(X, bn.vars):
        factors.append()


def order(X, vars):
    """
    Returns ordering for the variables.
    """
    order_list = []

    for var in vars:
        if var != X:
            order_list.append(var)
    order_list.append(X)

    return order_list


def make_factor(var, e):
    pass


def pointwise_product(factors):
    pass


def normalize(product):
    pass


if __name__ == "__main__":
    bayes_instance = BayesNetworkClass()

    # initialize nodes and edges list
    nodes = ["burglary", "earthquake", "JohnCalls", "MaryCalls", "alarm"]
    edges = [
        ("burglary", "alarm"),
        ("earthquake", "alarm"),
        ("alarm", "JohnCalls"),
        ("alarm", "MaryCalls"),
    ]

    # create graph
    bayesNet = DiscreteBayesianNetwork()
    bayesNet == bayes_instance.initialize_network(bayesNet, nodes, edges)

    # add CPDs for each node
    cpd_burglary = bayes_instance.add_cpd_to_node(
        bayesNet, "burglary", [[0.999], [0.001]], None, None
    )

    cpd_earthquake = bayes_instance.add_cpd_to_node(
        bayesNet, "earthquake", [[0.998], [0.002]], None, None
    )

    cpd_alarm = bayes_instance.add_cpd_to_node(
        bayesNet,
        "alarm",
        [[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
        ["burglary", "earthquake"],
        [2, 2],
    )

    cpd_john_calls = bayes_instance.add_cpd_to_node(
        bayesNet, "JohnCalls", [[0.95, 0.10], [0.05, 0.90]], ["alarm"], [2]
    )

    cpd_mary_calls = bayes_instance.add_cpd_to_node(
        bayesNet, "MaryCalls", [[0.99, 0.30], [0.01, 0.70]], ["alarm"], [2]
    )

    # check if model is correctly added
    bayesNet.check_model()
    print("Model is correct.")
