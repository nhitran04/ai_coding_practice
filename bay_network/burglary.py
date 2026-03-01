"""
You have a new burglar alarm installed at home. It is fairly reliable at detecting a burglary, but is occassionally
set off by minor earthquakes. You also have two neighbors, John and Mary. John is a heavy sleeper and will only call you if he hears the alarm,
but sometimes confuses the telephone ringing with the alarm. Mary, on the other hand, likes rather loud music and often misses the alarm altogether.
Given the evidence of who has or has not called, we would like to estimate the probability of a burglary.
"""

from bayes_class import BayesNetworkClass
from pgmpy.models import DiscreteBayesianNetwork


class BurglaryClass:
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
        bayesNet = bayes_instance.initialize_network(bayesNet, nodes, edges)

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

        # calculate probabilities
        print("burglary", bayes_instance.find_probability(bayesNet, ["burglary"], None))
        print(
            "earthquake",
            bayes_instance.find_probability(bayesNet, ["earthquake"], None),
        )
        print(
            "alarm | burglary, earthquake",
            bayes_instance.find_probability(
                bayesNet, ["alarm"], {"burglary": 1, "earthquake": 1}
            ),
        )
        print(
            "alarm | burglary, ~earthquake",
            bayes_instance.find_probability(
                bayesNet, ["alarm"], {"burglary": 1, "earthquake": 0}
            ),
        )
        print(
            "alarm | ~burglary, earthquake",
            bayes_instance.find_probability(
                bayesNet, ["alarm"], {"burglary": 0, "earthquake": 1}
            ),
        )
        print(
            "alarm | ~burglary, ~earthquake",
            bayes_instance.find_probability(
                bayesNet, ["alarm"], {"burglary": 0, "earthquake": 0}
            ),
        )
        print(
            "JohnCalls | alarm",
            bayes_instance.find_probability(bayesNet, ["JohnCalls"], {"alarm": 1}),
        )
        print(
            "JohnCalls | ~alarm",
            bayes_instance.find_probability(bayesNet, ["JohnCalls"], {"alarm": 0}),
        )
        print(
            "MaryCalls | alarm",
            bayes_instance.find_probability(bayesNet, ["MaryCalls"], {"alarm": 1}),
        )
        print(
            "MaryCalls | ~alarm",
            bayes_instance.find_probability(bayesNet, ["MaryCalls"], {"alarm": 0}),
        )
