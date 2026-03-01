"""
You have a new burglar alarm installed at home. It is fairly reliable at detecting a burglary, but is occassionally
set off by minor earthquakes. You also have two neighbors, John and Mary. John is a heavy sleeper and will only call you if he hears the alarm,
but sometimes confuses the telephone ringing with the alarm. Mary, on the other hand, likes rather loud music and often misses the alarm altogether.
Given the evidence of who has or has not called, we would like to estimate the probability of a burglary.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, List, Optional, Tuple


def initialize_network(
    nodes: List[str], edges: List[Tuple[str, str]]
) -> DiscreteBayesianNetwork:
    """
    Initializes the Bayesian Network with given nodes and edges.
    """
    bayesNet = DiscreteBayesianNetwork()

    for node in nodes:
        bayesNet.add_node(node)

    for parent, child in edges:
        bayesNet.add_edge(parent, child)

    return bayesNet


def add_cpd_to_node(
    node: str,
    values: List[List[float]],
    evidence: Optional[List[str]],
    evidence_card: Optional[List[int]],
) -> TabularCPD:
    """
    Adds a CPD to the Bayesian Network.
    """
    cpd = TabularCPD(
        node, len(values), values=values, evidence=evidence, evidence_card=evidence_card
    )
    bayesNet.add_cpds(cpd)

    return cpd


def find_probability(variables: List[str], evidence: Dict[str, int]) -> float:
    """
    Finds the probability of given variables with evidence.
    """
    solver = VariableElimination(bayesNet)
    result = solver.query(variables=variables, evidence=evidence)
    return result.values[1]


if __name__ == "__main__":
    # initialize nodes and edges list
    nodes = ["burglary", "earthquake", "JohnCalls", "MaryCalls", "alarm"]
    edges = [
        ("burglary", "alarm"),
        ("earthquake", "alarm"),
        ("alarm", "JohnCalls"),
        ("alarm", "MaryCalls"),
    ]

    # create graph
    bayesNet = initialize_network(nodes, edges)

    # add CPDs for each node
    cpd_burglary = add_cpd_to_node("burglary", [[0.999], [0.001]], None, None)
    cpd_earthquake = add_cpd_to_node("earthquake", [[0.998], [0.002]], None, None)
    cpd_alarm = add_cpd_to_node(
        "alarm",
        [[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
        ["burglary", "earthquake"],
        [2, 2],
    )
    cpd_john_calls = add_cpd_to_node(
        "JohnCalls", [[0.95, 0.10], [0.05, 0.90]], ["alarm"], [2]
    )
    cpd_mary_calls = add_cpd_to_node(
        "MaryCalls", [[0.99, 0.30], [0.01, 0.70]], ["alarm"], [2]
    )

    # check if model is correctly added
    bayesNet.check_model()
    print("Model is correct.")

    # probability there is a burglary
    print("burglary", find_probability(["burglary"], None))

    # probability there is an earthquake
    print("earthquake", find_probability(["earthquake"], None))

    # probability the alarm rings given that both a burglary and earthquake occur
    print(
        "alarm | burglary, earthquake",
        find_probability(["alarm"], {"burglary": 1, "earthquake": 1}),
    )

    # probability the alarm rings given a burglary occurs but not an earthquake
    print(
        "alarm | burglary, ~earthquake",
        find_probability(["alarm"], {"burglary": 1, "earthquake": 0}),
    )

    # probability the alarm rings given an earthquake occurs but not a burglary
    print(
        "alarm | ~burglary, earthquake",
        find_probability(["alarm"], {"burglary": 0, "earthquake": 1}),
    )
    # probability the alarm rings given neither the earthquake nor burglary occur
    print(
        "alarm | ~burglary, ~earthquake",
        find_probability(["alarm"], {"burglary": 0, "earthquake": 0}),
    )

    # probability that john calls given the alarm rings
    print("JohnCalls | alarm", find_probability(["JohnCalls"], {"alarm": 1}))

    # probability that john calls given the alarm does not ring
    print("JohnCalls | ~alarm", find_probability(["JohnCalls"], {"alarm": 0}))

    # probability that mary calls given the alarm rings
    print("MaryCalls | alarm", find_probability(["MaryCalls"], {"alarm": 1}))

    # probability that mary calls given the alarm does not rings
    print("MaryCalls | ~alarm", find_probability(["MaryCalls"], {"alarm": 0}))
