"""
Tutorial: Create and Inference Bayesian Networks using Pgmpy with Example
http://anmolkapoor.in/2019/05/05/Inference-Bayesian-Networks-Using-Pgmpy-With-Social-Moderator-Example/

Lets consider an example, where a social media website wish to moderate content on the
site and suspends bad user accounts. For this they would like us to create a statistical
moderator that can take the preemtive measure based on information given. Lets assume
we have following information:

Nodes:
    M : A prediction from a ML model that can read the content and give a score
        (probability) that this content should be flagged.
    U : Another User flags the content.
    B : The account was suspended before for any bad content.
    R : Score (Probability) that the content should be removed from the platform.
    S : Score (Probability) that account should be suspended.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

if __name__ == "__main__":
    # initialize nodes and edges list
    nodes = ["M", "U", "R", "B", "S"]
    edges = [("M", "R"), ("U", "R"), ("B", "R"), ("B", "S"), ("R", "S")]

    # create the acyclic directed graph for network
    bayesNet = DiscreteBayesianNetwork()

    for node in nodes:
        bayesNet.add_node(node)

    for parent, child in edges:
        bayesNet.add_edge(parent, child)

    # add CPDs for each node
    cpd_A = TabularCPD("M", 2, values=[[0.95], [0.05]])
    cpd_U = TabularCPD("U", 2, values=[[0.85], [0.15]])
    cpd_H = TabularCPD("B", 2, values=[[0.90], [0.10]])

    cpd_S = TabularCPD(
        "S",
        2,
        values=[[0.98, 0.88, 0.95, 0.6], [0.02, 0.12, 0.05, 0.40]],
        evidence=["R", "B"],
        evidence_card=[2, 2],
    )

    cpd_R = TabularCPD(
        "R",
        2,
        values=[
            [0.96, 0.86, 0.94, 0.82, 0.24, 0.15, 0.10, 0.05],
            [0.04, 0.14, 0.06, 0.18, 0.76, 0.85, 0.90, 0.95],
        ],
        evidence=["M", "B", "U"],
        evidence_card=[2, 2, 2],
    )

    bayesNet.add_cpds(cpd_A, cpd_U, cpd_H, cpd_S, cpd_R)

    # check if model is correctly added
    bayesNet.check_model()
    print("Model is correct.")

    # find the probability of conent should be removed from the platform
    solver = VariableElimination(bayesNet)
    result = solver.query(variables=["R"])
    print("R", result.values[1])

    # find the probability of content should be removed from platform given our ML model flags it
    result = solver.query(variables=["R"], evidence={"M": 1})
    print("R | M", result.values[1])

    # find the probability of account should be suspended given it was suspended before
    result = solver.query(variables=["S"], evidence={"B": 1})
    print("S | B", result.values[1])

    # find dependencies between variables
    bayesNet.get_independencies()
    print("Independencies: ", bayesNet.get_independencies())

    # completed
    print("Completed.")
