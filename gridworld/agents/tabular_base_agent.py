"""
base agent class, and the unconstrained Policy Improvement for tabular CMDPs
"""

from copy import deepcopy
import numpy as np
import cvxpy as cp
from gridworld.core.utils import default_termination, direct_policy_evaluation

class Agent:
    """
    base class
    """
    def __init__(self, **kwargs):
        self.params = deepcopy(kwargs)

    def set_logger(self, logger):
        self.logger = logger

    def make_policy_iteration_operator(self, **args):
        raise NotImplementedError

    def log(self):
        pass


class UnconstPIAgent(Agent):
    """
    Does regular policy improvement for based via solving LP at each step, without
        taking constraints into account

    """
    def __init__(self,
                 termination_condition,
                 **kwargs):
        super().__init__(**kwargs)
        self.termination_condition = termination_condition
        self.coeff_list = [(None, None)]
        self._name = "Unconstrained_PI"

    # Policy Iteration operator (just uses R)
    def make_policy_iteration_operator(self, P, R, discount, **kwargs):
        """
        P: sat
        R sa
        returns a operator that does 1 step of policy improvement
        via the Linear Programming formulation of the PI

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

            # state-wise PI
            for s in range(nstates):
                # construct local pi
                pi = cp.Variable(shape=(1, nactions))
                obj = cp.Maximize(cp.sum(cp.multiply(pi, Q[[s]])))  # <Q(s,.), \pi(.|s)>

                # add lower bound constraint
                constr = [pi[0] >= 0]

                # define the probability constraints
                constr += [cp.sum(pi[0]) == 1]

                # solve
                prob = cp.Problem(obj, constr)
                prob.solve()

                # normalize the policy policy (if needed)
                new_policy = pi.value

                # copy the solution for this state
                soln_pi[s] = new_policy[0]

            return soln_pi

        return lp_policy_iteration_operator

class RewardShapingPIAgent(Agent):
    """
    Does regular policy improvement for reshaped rewards, without
        taking constraints into account

    """
    def __init__(self,
                 termination_condition,
                 coeff_list,
                 **kwargs):
        super().__init__(**kwargs)
        self.termination_condition = termination_condition
        self.coeff_list = coeff_list
        self._name = "Reshaped_PI"


    def make_policy_iteration_operator(self, P, R, C, discount, coeffs, **kwargs):
        """
        P: sat
        R: sa
        C: sa

        coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0
        """
        nstates = R.shape[0]
        nactions = R.shape[1]


        def reward_shaping_pi_operator(policy):
            # compute Q using direct policy evaluation
            vR = direct_policy_evaluation(P, R, discount, policy)
            QR = R + discount * np.einsum('sat,t -> sa', P, vR)

            # for the cost
            vC = direct_policy_evaluation(P, C, discount, policy)
            QC = C + discount * np.einsum('sat,t -> sa', P, vC)

            # create the objective
            QL = coeffs[0] * QR - coeffs[1] * QC

            # placeholder policy
            soln_pi = np.zeros((nstates, nactions))

            # add state based constraints
            for s in range(nstates):
                # OPT
                pi = cp.Variable(shape=(1, nactions))  # prob for each action in each state
                obj = cp.Maximize(cp.sum(cp.multiply(pi, QL[[s]])))  # <Q_L(s,.), \pi(.|s)>

                # add lower bound constraint
                constr = [pi[0] >= 0.0]

                # define the probability constraints
                constr += [cp.sum(pi[0]) == 1.0]

                # solve
                prob = cp.Problem(obj, constr)
                prob.solve()

                new_policy = pi.value

                # copy the solution for this state
                soln_pi[s] = new_policy[0]

            return soln_pi

        return reward_shaping_pi_operator


class ReshapedAdvantagePIAgent(Agent):
    """
    Does regular policy improvement for reshaped rewards, along with
    the advantage constraints in the MLE
    """
    def __init__(self,
                 termination_condition,
                 coeff_list,
                 **kwargs):
        super().__init__(**kwargs)
        self.termination_condition = termination_condition
        self.coeff_list = coeff_list
        self._name = "Reshaping_Adv_PI"


    def make_policy_iteration_operator(self, P, R, C, discount, baseline, coeffs, **kwargs):
        """
        P: sat
        R: sa
        C: sa

        coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0
        """
        nstates = R.shape[0]
        nactions = R.shape[1]

        # calculate the estimates for the baseline policy
        vR_b = direct_policy_evaluation(P, R, discount, baseline)
        QR_b = R + discount * np.einsum('sat,t -> sa', P, vR_b)
        AR_b = QR_b - vR_b.reshape((nstates, 1))

        vC_b = direct_policy_evaluation(P, C, discount, baseline)
        QC_b = C + discount * np.einsum('sat,t -> sa', P, vC_b)
        AC_b = QC_b - vC_b.reshape((nstates, 1))

        def adv_q_learning_operator(policy):
            # compute Q using direct policy evaluation
            # for the reward
            vR = direct_policy_evaluation(P, R, discount, policy)
            QR = R + discount * np.einsum('sat,t -> sa', P, vR)

            # for the cost
            vC = direct_policy_evaluation(P, C, discount, policy)
            QC = C + discount * np.einsum('sat,t -> sa', P, vC)

            # create the objective
            QL = coeffs[0] * QR - coeffs[1] * QC

            # placeholder policy
            soln_pi = np.zeros((nstates, nactions))

            # add state based constraints
            for s in range(nstates):
                # OPT
                pi = cp.Variable(shape=(1, nactions))  # prob for each action in each state
                obj = cp.Maximize(cp.sum(cp.multiply(pi, QL[[s]])))  # <Q_L(s,.), \pi(.|s)>

                # add lower bound constraint
                constr = [pi[0] >= 0.0]

                # define the probability constraints
                constr += [cp.sum(pi[0]) == 1.0]

                # Advantage based constraints
                constr += [cp.sum(cp.multiply(pi, AR_b[[s]])) >= 0.0]  # R
                constr += [cp.sum(cp.multiply(pi, AC_b[[s]])) <= 0.0]  # C

                # solve
                prob = cp.Problem(obj, constr)
                prob.solve()

                new_policy = pi.value

                # copy the solution for this state
                soln_pi[s] = new_policy[0]

            return soln_pi

        return adv_q_learning_operator


