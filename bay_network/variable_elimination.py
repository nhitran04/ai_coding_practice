"""
The variable elimination algorithm for exact inference in Bayesian networks.
"""


def elimination_ask(X, e, bn):
    """
    Returns the probability distribution of the query variable X given the observed values e in the Bayesian network bn with variables vars.
    """
    factors = []
    for var in bn.vars:
        pass


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
    pass
