import numpy as np
import cvxpy as cp

from gridworld.core.utils import direct_policy_evaluation, bounded_successive_approximation, default_termination


class ValueFunction(object):
    """
    From: https://github.com/clvoloshin/constrained_batch_policy_learning/blob/master/value_function.py
    """

    def __init__(self):
        '''
        Utility DS to store the previous values and policies
        '''
        self.prev_values = []

    def append(self, value):
        self.prev_values.append(value)

    def avg(self, append_zero=False):
        if append_zero:
            return np.hstack([np.mean(self.prev_values, 0), np.array([0])])
        else:
            return np.mean(self.prev_values, 0)

    def last(self, append_zero=False):
        if append_zero:
            return np.hstack([self.prev_values[-1], np.array([0])])
        else:
            return np.array(self.prev_values[-1])


# Regular Policy Iteration operator (just uses R)

def make_cost_policy_iteration_operator(P, R, discount, **kwargs):
    """
    P: sat
    R: sa
    returns a operator that does 1 step of policy improvement
    via the Linear Programming formulation of the PI

    Imp: minimizes the cost!

    :param P: transition matrix
    """
    nstates = R.shape[0]
    nactions = R.shape[1]

    def lp_policy_iteration_operator(policy):
        # calculate the q-values
        v = direct_policy_evaluation(P, R, discount, policy)
        Q = R + discount * np.einsum('sat,t -> sa', P, v)

        # final policy placeholder
        soln_pi = np.zeros((nstates, nactions))

        # greedy update
        soln_pi[np.arange(nstates), np.argmin(Q, axis=1)] = 1.0
        return soln_pi

    return lp_policy_iteration_operator


def cost_policy_iteration(P, R, discount, max_iters=25):
    """
    P: sat
    R: sa
    does policy iteration, starting from a random policy and doing 1-step PI based on LP
    """
    # create the operator
    pi_operator = make_cost_policy_iteration_operator(P, R, discount)

    # create random pi
    nstates = R.shape[0]
    nactions = R.shape[1]

    random_policy = np.ones((nstates, nactions)) / nactions

    pi_solution = bounded_successive_approximation(random_policy,
                                                   operator=pi_operator,
                                                   termination_condition=default_termination,
                                                   max_limit=max_iters, )

    return pi_solution


