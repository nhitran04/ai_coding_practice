from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, List, Optional, Tuple
from pgmpy.models import DiscreteBayesianNetwork


class BayesNetworkClass:
    """
    Provides methods needed for Bayesian network.
    """

    @staticmethod
    def initialize_network(
        bayesNet: DiscreteBayesianNetwork,
        nodes: List[str],
        edges: List[Tuple[str, str]],
    ) -> DiscreteBayesianNetwork:
        """
        Initializes the Bayesian Network with given nodes and edges.
        """
        for node in nodes:
            bayesNet.add_node(node)

        for parent, child in edges:
            bayesNet.add_edge(parent, child)

        return bayesNet

    @staticmethod
    def add_cpd_to_node(
        bayesNet: DiscreteBayesianNetwork,
        node: str,
        values: List[List[float]],
        evidence: Optional[List[str]],
        evidence_card: Optional[List[int]],
    ) -> TabularCPD:
        """
        Adds a CPD to the Bayesian Network.
        """
        cpd = TabularCPD(
            node,
            len(values),
            values=values,
            evidence=evidence,
            evidence_card=evidence_card,
        )
        bayesNet.add_cpds(cpd)
        print("===== CPD for node" + node + " =====")
        print(cpd)
        return cpd

    @staticmethod
    def find_probability(
        bayesNet: DiscreteBayesianNetwork,
        variables: List[str],
        evidence: Dict[str, int],
    ) -> float:
        """
        Finds the probability of given variables with evidence.
        """
        solver = VariableElimination(bayesNet)
        result = solver.query(variables=variables, evidence=evidence)
        return result.values[1]
