"""
Contains the code for the scaling experiments in Appendix D
"""
import cvxpy as cp
import numpy as np

from scipy import stats
from math import log,sqrt

from gridworld.core.utils import direct_policy_evaluation, default_termination, bounded_successive_approximation, estimate_model
from gridworld.agents.tabular_base_agent import Agent



def multi_cost_cmdp_dual_lp(P, R, C_list, discount, d0,
                            initial_distribution):
    """

    :param P: transition matrix of shape [|S|,|A|,|S|]
    :param R: shape [|S|,|A|]
    :param C: list now
    :param discount: gamma
    :param d0: the limit of CMDP
    :param initial_distribution: mu vector

    :return: a tuple of [state_dist_values, cost_at_optimal_solution, corresponding_optimal_policy]
    """
    nstates, nactions = R.shape

    p = cp.Variable(shape=(nstates, nactions), nonneg=True)
    # Nonneg flags >=0 and takes care of floating precision errors

    obj = cp.Maximize(cp.sum(cp.multiply(p, R)))

    # add lower bound constraint / redundant because of nonneg flag
    constr = [p >= 0]

    # define constrain for each x'
    # flow conservation constraints. for each s',
    # \sum_{s, a} x(s, a) (1_{s = s'} - \gamma * T(s, a, s')) = \alpha(s')
    constr += [cp.sum(p[s]) - discount * cp.sum(cp.multiply(P[:, :, s], p)) == initial_distribution[s] for s in
               range(nstates)]

    # add the traj cost constraint also
    for C in C_list:
        constr += [cp.sum(cp.multiply(p, C)) <= d0]

    # sovle
    prob = cp.Problem(obj, constr)
    # Let CVXPY chose the best solver for this problem
    prob.solve(verbose=False)

    pi_opt = p.value
    pi_opt = pi_opt / pi_opt.sum(axis=1)[:, None]

    cum_cost_list = []
    for C in C_list:
        cum_cost = np.sum(np.asarray(p.value) * C)
        cum_cost_list.append(cum_cost)

    return prob.value, cum_cost_list, pi_opt


class MultiConstSPIBBAgent(Agent):
    """
    The agent based on the S-OPT equation in the draft that scales to multiple cost functions
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

    def make_policy_iteration_operator(self, P, R, C_list, discount, baseline, error_fn, epsilon, coeffs,):
        """
        P: sat
        R: sa
        C: list of [sa]

        coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0

        epsilon: if eps = np.inf, then (\pi_b,e_Q,eps) constraint is not enforced
        """
        nstates = R.shape[0]
        nactions = R.shape[1]

        # calculate the estimates for the baseline policy
        vR_b = direct_policy_evaluation(P, R, discount, baseline)
        QR_b = R + discount * np.einsum('sat,t -> sa', P, vR_b)
        AR_b = QR_b - vR_b.reshape((nstates, 1))

        # scale to constraints
        num_cost_funcs = len(C_list)
        vC_b_list = []
        QC_b_list = []
        AC_b_list = []

        for cost_idx in range(num_cost_funcs):
            # select the cost func
            C = C_list[cost_idx]

            # calculate the estimates
            vC_b = direct_policy_evaluation(P, C, discount, baseline)
            QC_b = C + discount * np.einsum('sat,t -> sa', P, vC_b)
            AC_b = QC_b - vC_b.reshape((nstates, 1))

            # append them to the list
            vC_b_list.append(vC_b)
            QC_b_list.append(QC_b)
            AC_b_list.append(AC_b)

        def constrained_spibb_policy_iteration_operator(policy):

            # compute Q using direct policy evaluation
            # for the reward
            vR = direct_policy_evaluation(P, R, discount, policy)
            QR = R + discount * np.einsum('sat,t -> sa', P, vR)

            # for the cost
            vC_list = []
            QC_list = []

            for cost_idx in range(num_cost_funcs):
                C = C_list[cost_idx]
                vC = direct_policy_evaluation(P, C, discount, policy)
                QC = C + discount * np.einsum('sat,t -> sa', P, vC)
                # append
                vC_list.append(vC)
                QC_list.append(QC)

            # create the objective
            # club the costs
            QL = coeffs[0] * QR
            for cost_idx in range(num_cost_funcs):
                QL -= (coeffs[1 + cost_idx] * QC_list[cost_idx])

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
                # wrt R
                constr += [cp.sum(cp.multiply(pi, AR_b[[s]])) >= 0.0]

                # wrt C
                for cost_idx in range(num_cost_funcs):
                    AC_b = AC_b_list[cost_idx]
                    constr += [cp.sum(cp.multiply(pi, AC_b[[s]])) <= 0.0]

                # Add another constraint based on correction index that preserves the
                # value of the baseline policy for that index
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



def make_hopt_adv_operator(P, R, C_list, discount, baseline, coeffs,):
    """
    IMP:
    P: sat
    R: sa
    C: list of [sa]

    coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0

    epsilon: if eps = np.inf, then (\pi_b,e_Q,eps) constraint is not enforced
    """
    nstates = R.shape[0]
    nactions = R.shape[1]

    # calculate the estimates for the baseline policy
    vR_b = direct_policy_evaluation(P, R, discount, baseline)
    QR_b = R + discount * np.einsum('sat,t -> sa', P, vR_b)
    AR_b = QR_b - vR_b.reshape((nstates, 1))

    # scale to constraints
    num_cost_funcs = len(C_list)
    vC_b_list = []
    QC_b_list = []
    AC_b_list = []

    for cost_idx in range(num_cost_funcs):
        # select the cost func
        C = C_list[cost_idx]

        # calculate the estimates
        vC_b = direct_policy_evaluation(P, C, discount, baseline)
        QC_b = C + discount * np.einsum('sat,t -> sa', P, vC_b)
        AC_b = QC_b - vC_b.reshape((nstates, 1))

        # append them to the list
        vC_b_list.append(vC_b)
        QC_b_list.append(QC_b)
        AC_b_list.append(AC_b)

    def adv_q_learning_operator(policy):

        # compute Q using direct policy evaluation
        # for the reward
        vR = direct_policy_evaluation(P, R, discount, policy)
        QR = R + discount * np.einsum('sat,t -> sa', P, vR)

        # for the cost
        vC_list = []
        QC_list = []

        for cost_idx in range(num_cost_funcs):
            C = C_list[cost_idx]
            vC = direct_policy_evaluation(P, C, discount, policy)
            QC = C + discount * np.einsum('sat,t -> sa', P, vC)
            # append
            vC_list.append(vC)
            QC_list.append(QC)

        # create the objective
        # club the costs
        QL = coeffs[0] * QR
        for cost_idx in range(num_cost_funcs):
            QL -= (coeffs[1 + cost_idx] * QC_list[cost_idx])

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
            # wrt R
            constr += [cp.sum(cp.multiply(pi, AR_b[[s]])) >= 0.0]

            # wrt C
            for cost_idx in range(num_cost_funcs):
                AC_b = AC_b_list[cost_idx]
                constr += [cp.sum(cp.multiply(pi, AC_b[[s]])) <= 0.0]


            # solve
            prob = cp.Problem(obj, constr)
            prob.solve()

            new_policy = pi.value

            # copy the solution for this state
            soln_pi[s] = new_policy[0]

        return soln_pi

    return adv_q_learning_operator



class MultiCostISEstimator:
    """
    The multiple objective importance sampling estimator class version.
    Contains only DR estimates
    Check the tabular_hcpi_agents.py for more details on the original class.

    It estimates the return of a target policy pi_t
    given a set of trajectories generated by a baseline policy pi_b
    The available estimator are :
    - importance sampling;
    - per decision importance sampling;
    """
    def __init__(self, gamma: float,
                 pi_b: np.ndarray,
                 pi_t: np.ndarray,
                 estimator_type: str,
                 R = None,
                 C_list = None,
                 P_hat = None ,
                 list_format = True):
        """

        :param gamma: discount factor
        :param pi_b: the baseline policy
        :param pi_t: the target policy
        :param estimator_type: the kind of IS algorithm the estimator uses
        :param R: Reward function
        :param C: Constraint function
        :param list_format: whether to return in a list or return a scalar
        """
        self.estimator_dict = {"doubly_robust": self.doubly_robust,
                              }
        self.gamma=gamma
        self.estimator_type = estimator_type
        self.pi_b = pi_b
        self.pi_t = pi_t
        self.list_format = list_format

        # we need these for DR estimator
        self.R = R
        self.C_list = C_list
        self.P_hat = P_hat

    def __call__(self, trajectories):
        """
        this way no need to create an object? the main method
        :param trajectories:
        :return:
        """
        return self.estimator_dict[self.estimator_type](trajectories)


    def doubly_robust(self, trajectories):
        """
        As implemented in Jiang and Li, 2015;
        Make use of a control variate build from an approximate
        model of the MDP
        :param trajectories: a set of trajectories
        :return: an estimate of the return
        """

        if self.list_format:
            R_estimate, C_estimate = self.compute_estimare_dr(trajectories)
            return R_estimate, C_estimate
        else:
            # We used the 2-fold DR as model fitting
            raise Exception("not implemented yet")


    def compute_estimare_dr(self, trajectories_eval, is_list=True):
        R_estimate = []
        C_estimate = {}
        for cost_idx in range(len(self.C_list)):
            C_estimate[cost_idx] = []


        # compute the V_hat and Q_hat for the pi_t
        # calculate the estimates for the target policy
        # R
        rV_hat = direct_policy_evaluation(self.P_hat, self.R, self.gamma, self.pi_t)
        rQ_hat = self.R + self.gamma * np.einsum('sat,t -> sa', self.P_hat, rV_hat)

        cV_hat_list = []
        cQ_hat_list = []

        for cost_idx in range(len(self.C_list)):
            # C
            cV_hat = direct_policy_evaluation(self.P_hat, self.C_list[cost_idx], self.gamma, self.pi_t)
            cQ_hat = self.C_list[cost_idx] + self.gamma * np.einsum('sat,t -> sa', self.P_hat, cV_hat)
            # append to lists
            cV_hat_list.append(cV_hat)
            cQ_hat_list.append(cQ_hat)

        for trajectory in trajectories_eval:
            r_estimate_trajectory = 0
            c_estimate_trajectory = np.zeros(len(self.C_list))

            for transition in trajectory[::-1]:
                state, action, reward, cost, next_state = transition

                # R
                r_estimate_trajectory = int(rV_hat[state]) + \
                        self.pi_t[state, action] / self.pi_b[state, action]* (reward +
                                              self.gamma * r_estimate_trajectory-int(rQ_hat[state, action]))

                # C
                for cost_idx in range(len(self.C_list)):
                    c_estimate_trajectory[cost_idx] = int(cV_hat_list[cost_idx][state]) + \
                                        self.pi_t[state, action] / self.pi_b[state, action] * ( cost[cost_idx] +
                                                self.gamma * c_estimate_trajectory[cost_idx] - int(cQ_hat_list[cost_idx][state, action]))

            if is_list:
                R_estimate.append(r_estimate_trajectory)
                for cost_idx in range(len(self.C_list)):
                    C_estimate[cost_idx].append(c_estimate_trajectory[cost_idx])

        return R_estimate, C_estimate


class MultiCostHCPIAgent(Agent):
    """
    The scaling with $d$ H-OPT agent method
    """
    def __init__(self,
                 lower_bound_strategy: str,
                 estimator_type : str,
                 coeff_list=[],
                 num_costs = 1,
                 training_size = 0.7,
                 nb_steps=5000,
                 ):
        """

        :param gamma: discount
        :param lower_bound_strategy: ??
        :param confidence: delta in student t-test
        :param estimator_type: the kind of IS estimator to use
        :param rho_min: for normalizing thes return from IS
        :param rho_max: for normalizing the return from IS
        :param nb_steps: number of steps for the q-learning updates
        """

        # super().__init__()
        self.strategy_dict = { #"CI": self.confidence_interval_based,
                              "student_t_test": self.student_t_test,
                             }

        self.lower_bound_strategy = lower_bound_strategy
        self.estimator_type = estimator_type
        self.training_size = training_size
        self.nb_steps = nb_steps
        self.coeff_list = coeff_list
        self.num_costs = num_costs

        # name takes into account the IS sampling and LB schemes
        self._name = "H_OPT" + "-" + estimator_type + "-"

    def __call__(self, trajectories):
        return self.strategy_dict[self.lower_bound_strategy](trajectories)

    def student_t_test(self, trajectories, confidence, estimator,
                       R_min, R_max,
                       C_min, C_max,
                       ):
        """
        Warning !! This method relies on the assumption that the return is normally distributed

        :param trajectories: a batch of trajectories
        :param confidence: the delta parameter
        :param estimator: the off policy evaluation estimator objec

        :return: a lower bound on the estimate return
        """

        # estimate the returns
        R_list_estimates, C_list_estimates = estimator(trajectories)

        # normalize
        R_list_estimates = self.normalize_return(R_list_estimates, x_min=R_min, x_max=R_max, )

        for cost_idx in range(self.num_costs):
            C_list_estimates[cost_idx] = self.normalize_return(C_list_estimates[cost_idx], x_min=C_min, x_max=C_max, )

        # t-test
        # Sec 2.4 HCPI (Thomas 2015)
        n = len(R_list_estimates)
        # w.r.t. R
        R_estimated_return = np.mean(R_list_estimates)
        R_sigma = np.sqrt(1./(n-1) * np.sum(np.square(np.array(R_list_estimates) - R_estimated_return)))
        R_lower_bound = R_estimated_return - R_sigma/sqrt((n-1)) * stats.t.ppf(1 - confidence, n-1)
        # undo the normalization here
        R_lower_bound = R_lower_bound * (R_max - R_min) + R_min

        # w.r.t. C
        C_upper_bound_list = []

        for cost_idx in range(self.num_costs):
            c_estimated_return = np.mean(C_list_estimates[cost_idx])
            c_sigma = np.sqrt(1. / (n - 1) * np.sum(np.square(np.array(C_list_estimates[cost_idx]) - c_estimated_return)))
            c_upper_bound = c_estimated_return + c_sigma / sqrt((n - 1)) * stats.t.ppf(1 - confidence, n - 1)
            # undo the normalization here
            c_upper_bound = c_upper_bound * (C_max - C_min) + C_min

            C_upper_bound_list.append(c_upper_bound)

        return R_lower_bound, C_upper_bound_list

    def normalize_return(self, list_estimate, x_min, x_max):
        """
        normalizes the return in [0,1]
        :param list_estimate:
        :return:
        """
        return [(x - x_min) / (x_max - x_min) for x in list_estimate]

    def compute_policy(self, trajectories, pi_b, confidence, coeffs,
                       R, C_list,
                       discount,
                       pib_R_est_performance, pib_C_est_performance_list,
                       R_min, R_max,
                       C_min, C_max,):
        """
        do the policy improvement here
        :return:
        """
        # do a train-test split
        training_index = int(self.training_size * len(trajectories))

        training_trajectories = trajectories[:training_index]
        testing_trajectories = trajectories[training_index:]

        # create the batch from training trajectories here
        batch = []
        for trajectory in training_trajectories:
            for [state, action, reward, cost, next_state] in trajectory:
                batch.append([state, action, reward, cost, next_state])

        nstates = R.shape[0]
        nactions = R.shape[1]
        P_hat = estimate_model(batch, nstates, nactions)


        # can use a PI operator also,
        approx_pi_operator = make_hopt_adv_operator(P=P_hat, R=R, C_list=C_list,
                                                    discount=discount,
                                                    baseline=pi_b,
                                                    coeffs=coeffs)

        # call Q-learning operator here
        # find the optimal pi
        pi_hat = bounded_successive_approximation(pi_b,
                                                  operator=approx_pi_operator,
                                                  termination_condition=default_termination,
                                                  max_limit=5,)

        # Regularization on the optimal policy
        regularization_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Evaluate all the policy and return the one with the higher lower bound
        best_R_lower_bound = - np.inf

        best_C_upper_bound_list = [+ np.inf] * self.num_costs

        current_best_policy = pi_hat
        current_regularization = 0

        # try different linear combinations
        for regularization_parameter in regularization_list:
            # new candidate policy
            current_pi = (1 - regularization_parameter) * pi_hat + \
                             regularization_parameter * pi_b

            # Create the estimator here
            estimator = MultiCostISEstimator(gamma=discount,
                                             pi_b=pi_b,
                                             pi_t=current_pi,
                                             estimator_type=self.estimator_type,
                                             R=R,
                                             C_list=C_list,
                                             P_hat=P_hat,
                                             list_format=True)

            # compute the bounds based on the test batch
            R_lb, C_ub_list = self.strategy_dict[self.lower_bound_strategy](testing_trajectories,
                                                                            confidence, estimator,
                                                                            R_min, R_max,
                                                                            C_min, C_max)

            # if the bounds are finite, only then use them, else skip
            if (not np.isfinite(R_lb)) or any(not np.isfinite(c_ub) for c_ub in C_ub_list):
                continue

            # accept the candidate if it has better performance w.r.t R and C
            if R_lb >= best_R_lower_bound and all(c_ub <= best_C_upper_bound_list[c_idx] for c_idx, c_ub in enumerate(C_ub_list)):
                current_best_policy = current_pi
                best_R_lower_bound = R_lb
                for c_idx, c_ub in enumerate(C_ub_list):
                    best_C_upper_bound_list[c_idx] = c_ub
                current_regularization = regularization_parameter


        # check if the best candidate passes the safety test
        # The first instance, uses the estimated returns using P_hat, and the mean performance of the baseline
        pi_sol, reg_sol = self.safety_test(current_best_policy, best_R_lower_bound, best_C_upper_bound_list,
                                           current_regularization, pi_b,
                                           pib_R_est_performance, pib_C_est_performance_list)

        return pi_sol, reg_sol

    def safety_test(self, pi_t, R_lb_target, C_ub_target_list,
                    current_regularization, pi_b,
                    R_perf_baseline, C_perf_baseline_list):
        """
        The No Solution Found (NSF) solution

        Essentially the safety test
        :return:
        """
        if R_lb_target >= R_perf_baseline and all(C_ub_target_list[c_idx] <= C_perf_baseline_list[c_idx] for c_idx in range(self.num_costs)):
            # passes the safety test
            return pi_t, current_regularization
        else:
            # return the original policy
            return pi_b, 1.0
