"""
Contains the implementation of MO-SPIBB (S-OPT) in the draft
"""
import cvxpy as cp
import numpy as np

from gridworld.core.utils import direct_policy_evaluation
from gridworld.agents.tabular_base_agent import Agent



class ConstSPIBBAgent(Agent):
    """
    The agent based on the S-OPT equation in the draft
    """
    def __init__(self,
                 termination_condition,
                 coeff_list,
                 **kwargs):
        """

        :param termintation_condition: the function that decides when to stop the iterative procedure
        :param coeff_list: the list of lambda_parameters to try search over
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.termination_condition = termination_condition
        self.coeff_list = coeff_list
        self._name = "S_OPT"

    def make_policy_iteration_operator(self, P, R, C, discount, baseline, error_fn, epsilon, coeffs,):
        """
        IMP:
        P: sat
        R: sa
        C: sa

        coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0

        epsilon: if eps = np.inf, then (\pi_b,e_Q,eps) constraint is not enforced
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

        def constrained_spibb_policy_iteration_operator(policy):

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

                # to find which err function estimates are okay to use in the optimization
                # only keep those that are < np.inf, and keep the rest to 0
                ok_err = np.zeros_like(error_fn[s])
                correction_idx = error_fn[s] < np.inf
                ok_err[correction_idx] = error_fn[s][correction_idx]

                # add the constraints now based on corrected ok_err
                if epsilon < np.inf:
                    constr += [cp.sum(cp.multiply(cp.abs(pi[0] - baseline[s]), ok_err)) <= epsilon]

                # Advantage based constraints
                constr += [cp.sum(cp.multiply(pi, AR_b[[s]])) >= 0.0]  # R
                constr += [cp.sum(cp.multiply(pi, AC_b[[s]])) <= 0.0]  # C

                # Add another constraint based on correction index that preserves the
                #   value of the baseline policy for that index
                for a in range(nactions):
                    if (error_fn[s][a] >= np.inf) and (epsilon < np.inf):
                        constr += [pi[0][a] == baseline[s][a]]

                # solve
                prob = cp.Problem(obj, constr)
                prob.solve()

                new_policy = pi.value

                # copy the solution for this state
                soln_pi[s] = new_policy[0]

            return soln_pi

        return constrained_spibb_policy_iteration_operator
