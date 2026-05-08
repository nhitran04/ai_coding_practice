'''
The variable elimination algorithm for exact inference in Bayesian networks.
'''


def elimination_ask(X, e, bn):
    '''
    Returns the probability distribution of the query variable X given the observed values e in the Bayesian network bn with variables vars.
    '''
    factors = []
    for var in bn.vars:
        pass
