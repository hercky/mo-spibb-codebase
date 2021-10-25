## Create the Lagrangian agent here!
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from gridworld.agents.tabular_base_agent import Agent
from gridworld.lag_baseline.utils import ValueFunction, cost_policy_iteration
from gridworld.core.utils import direct_policy_evaluation, bounded_successive_approximation
from gridworld.lag_baseline.exponentiated_gradient import ExponentiatedGradient


## Create the Lagrangian agent here!

class Lagrangian(Agent):
    """
    Agent based on Le at al (2109)
    """

    def __init__(self,
                 coeff_list,
                 n_iter,
                 norm_bound,
                 lr_eta,
                 dual_gap,
                 ):
        """

        :param termintation_condition: the function that decides when to stop the iterative procedure
        :param coeff_list: the list of lambda_parameters to try search over
        :param kwargs:
        """
        self.coeff_list = coeff_list
        self._name = "Lagrangian"

        # Hyper-params
        self.n_iter = n_iter  # hyper-param t in Line 2 of Alg 2
        self.lambda_bound = norm_bound  # hyper-param B input to Alg 2
        self.lr_eta = lr_eta  # hyper-param \eta input to Alg 2
        self.dual_gap = dual_gap  # hyper-param \omega in Line 13 in Alg 2

        # init the containters
        self.reset()

    def reset(self):
        """flushes all the values for a fresh run"""
        # init the containers for storing value estimates
        self.U_vals = ValueFunction()
        self.R_vals = ValueFunction()
        self.C_vals = ValueFunction()

        # perf in true env
        self.true_R_vals = ValueFunction()
        self.true_C_vals = ValueFunction()

        # plotting utils
        self.safe_log = []
        self.performance_log_R = []
        self.performance_log_C = []
        self.gap_log = []

    def make_policy_iteration_operator(self, P, R, C, discount,
                                       initial_distribution,
                                       coeffs,
                                       pi_baseline,
                                       P_star,
                                       plotting=True,
                                       ):
        """
        Algorithm 2 of Le et al (2019)

        Estimated mdp parameters
        P: sat
        R: sa
        C: sa
        P_star: sat

        P_star required for the true policy evaluation that

        coeffs: coefficients (\lambda_i) for each signal AND \lamba_i >= 0
        """
        # flush all the past info
        self.reset()

        nstates = R.shape[0]
        nactions = R.shape[1]

        # these are the tau, the new constraints!
        vR_pib_mhat = direct_policy_evaluation(P, - R, discount,
                                               pi_baseline)  # -ve becasue we are now dealing with costs
        pib_R_est_performance = sum(vR_pib_mhat * initial_distribution)

        vC_pib_mhat = direct_policy_evaluation(P, C, discount, pi_baseline)
        pib_C_est_performance = sum(vC_pib_mhat * initial_distribution)

        # tau = [R,C,0] (#constraints + 1)
        # the SPI constraints are on performance and the additional constraint is due to the bounded norm
        tau = [pib_R_est_performance, pib_C_est_performance, 0.]

        # initialize the online algo
        online_convex_algorithm = ExponentiatedGradient(self.lambda_bound, len(tau), self.lr_eta,
                                                        starting_lambda='uniform')

        # init lambda (Line 1, Alg 2)
        lambdas = []
        lambdas.append(online_convex_algorithm.get())

        # Loggging/debugging
        vR_pib_true = direct_policy_evaluation(P_star, R, discount, pi_baseline)  # +ve here, because just evaluating
        pib_R_true_performance = sum(vR_pib_true * initial_distribution)
        vC_pib_true = direct_policy_evaluation(P_star, C, discount, pi_baseline)
        pib_C_true_performance = sum(vC_pib_true * initial_distribution)

        # (Line 2, Alg 2)
        for t in tqdm(range(self.n_iter), disable=1 - plotting):
            # current_lambda
            lamb = lambdas[-1]

            # the user objective
            U = -1. * (coeffs[0] * R - coeffs[
                1] * C)  # negative because Le et al minimizing cost instead of maximize return

            # Do best response PI here and get new policy (Line 3)
            best_response_cost_fn = U + (lamb[0] * - R) + (lamb[1] * C)
            pi_t = cost_policy_iteration(P=P, R=best_response_cost_fn, discount=discount)

            # Do PE here (Line 4,5)
            U_pi_t_hat, R_pi_t_hat, C_pi_t_hat = self.update_vals(pi_t, P, U, R, C, discount, initial_distribution,
                                                                  P_star)

            # No need to compute pi_hat_t if we are already storing the values (Skip Line 6)

            # Take the averages here (Line 7)
            U_avg = self.U_vals.avg()
            R_avg = self.R_vals.avg()
            C_avg = self.C_vals.avg()

            # Get the avg lambda (Line 8)
            lamb_avg = np.mean(lambdas, 0)

            # Get the min over lagrangian (Lines 9, 10, 12)
            L_min = self.min_of_lagrangian_over_policy(lamb_avg, tau, P, U, R, C, discount, initial_distribution)

            # Compute the max over lagrangian (Line 11)
            G_avg = [R_avg, C_avg, 0]
            L_max = self.max_of_lagrangian_over_lambda(U_avg, G_avg, tau)

            # Check the duality gap here (Line 13)
            if L_max - L_min <= self.dual_gap:
                # Terminate the algorithm (Line 14)
                # As we are not storing all the policies so far, we just compute the avg of the true performance of the policies so far
                #                 true_R_perf = self.true_R_vals.avg()
                #                 true_C_perf = self.true_C_vals.avg()
                #                 return true_R_perf, true_C_perf
                break

            # Update the lambdas here (Line 15,16)
            G_last = np.array([R_pi_t_hat, C_pi_t_hat, 0.])
            gradient = G_last - tau
            lambda_t = online_convex_algorithm.run(gradient)

            # Append the updated lambda to the list
            lambdas.append(lambda_t)

            # Logging stuff
            is_safe = (self.true_R_vals.avg() >= pib_R_true_performance) and (
                        self.true_C_vals.avg() <= pib_C_true_performance)
            self.safe_log.append(is_safe)
            self.gap_log.append(L_max - L_min)
            self.performance_log_R.append(self.true_R_vals.avg())
            self.performance_log_C.append(self.true_C_vals.avg())
            if t % 50 == 0 and plotting:
                print(f"Iter: {t}, Gap: {L_max - L_min:.3f}, Safe: {is_safe}")

                # plot for loggging purpose
        if plotting:
            self.plot(pib_R_true_performance, pib_C_true_performance)

        # return the final performance
        true_R_perf = self.true_R_vals.avg()
        true_C_perf = self.true_C_vals.avg()
        true_failure_rate = 1. - np.mean(self.safe_log)

        return true_R_perf, true_C_perf, true_failure_rate

    def update_vals(self, pi, P, U, R, C, discount, initial_distribution, P_star):
        """
        do policy evalutation and update the values
        """
        # Do PE here
        # For user preferences based cost
        v_U_pi = direct_policy_evaluation(P, U, discount, pi)
        U_pi_hat = sum(v_U_pi * initial_distribution)

        # For R
        v_R_pi = direct_policy_evaluation(P, - R, discount, pi)  # -ve because we are now working with costs
        R_pi_hat = sum(v_R_pi * initial_distribution)

        # For C
        v_C_pi = direct_policy_evaluation(P, C, discount, pi)
        C_pi_hat = sum(v_C_pi * initial_distribution)

        # store them
        self.U_vals.append(U_pi_hat)
        self.R_vals.append(R_pi_hat)
        self.C_vals.append(C_pi_hat)

        # now calculate the real performance in the true environment (only for R and C)
        # For R
        exact_v_R_pi = direct_policy_evaluation(P_star, R, discount, pi)
        exact_R_pi = sum(exact_v_R_pi * initial_distribution)
        # For C
        exact_v_C_pi = direct_policy_evaluation(P_star, C, discount, pi)
        exact_C_pi = sum(exact_v_C_pi * initial_distribution)

        self.true_R_vals.append(exact_R_pi)
        self.true_C_vals.append(exact_C_pi)

        return U_pi_hat, R_pi_hat, C_pi_hat

    def min_of_lagrangian_over_policy(self, lamb_avg, tau, P, U, R, C, discount, initial_distribution):
        """
        Performs the steps in Line 9,10 and 12 of Alg 2
        """
        # Learn the pi_tilde (Line 9)
        avg_best_response_cost_fn = U + (lamb_avg[0] * - R) + (lamb_avg[1] * C)
        pi_tilde = cost_policy_iteration(P=P, R=avg_best_response_cost_fn, discount=discount)

        # Eval the pi_tilde (Line 10)
        # For L (user preferences based cost)
        v_U_pi_tilde = direct_policy_evaluation(P, U, discount, pi_tilde)
        U_pi_tilde_hat = sum(v_U_pi_tilde * initial_distribution)

        # For R
        v_R_pi_tilde = direct_policy_evaluation(P, - R, discount, pi_tilde)  # -ve because we are now working with costs
        R_pi_tilde_hat = sum(v_R_pi_tilde * initial_distribution)

        # For C
        v_C_pi_tilde = direct_policy_evaluation(P, C, discount, pi_tilde)
        C_pi_tilde_hat = sum(v_C_pi_tilde * initial_distribution)

        # Compute the L_min (Line 12)
        G_pi_tilde = np.array([R_pi_tilde_hat, C_pi_tilde_hat, 0.])
        L_min = U_pi_tilde_hat + np.dot(lamb_avg, (G_pi_tilde - tau))

        return L_min

    def max_of_lagrangian_over_lambda(self, U_avg, G_avg, tau):
        '''
        Line 11 of Alg 2

        The maximum of C(pi) + lambda^T (G(pi) - eta) over lambda is
        B*e_{k+1}, all the weight on the phantom index if G(pi) < eta for all constraints
        B*e_k otherwise where B is the l1 bound on lambda and e_k is the standard
        basis vector putting full mass on the constraint which is violated the most
        '''
        G_avg = np.array(G_avg)

        ## Calculate the max violating constraint here
        maximum = np.max(G_avg - tau)
        index = np.argmax(G_avg - tau)

        if maximum > 0:
            # if there is constraint that is being violated the most, put all weight on it
            lamb = self.lambda_bound * np.eye(1, len(tau), index).reshape(-1)
        else:
            # put all weight on the phantom lambda
            lamb = np.zeros(len(tau))
            lamb[-1] = self.lambda_bound

        # compute the lagrangian wrt this lambda
        L_max = U_avg + np.dot(lamb, (G_avg - tau))

        return L_max

    def plot(self, pib_R_true_performance, pib_C_true_performance):
        """
        plot how the training looks for debugging
        """
        # Create sub-plot for 4 plots
        # Top safety, duality gap
        # Bottom R_perf, C_perf (w/ baseline performance)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # num of iters
        x_range = np.arange(len(self.safe_log))

        safe_rate = np.zeros(len(self.safe_log))
        for i in range(len(self.safe_log)):
            safe_rate[i] = 1. - np.mean(self.safe_log[:i + 1])
        axes[0, 0].plot(x_range, safe_rate)
        axes[0, 0].set_ylabel("Failure rate")
        #         axes[0, 0].set_xlabel("iterations")
        axes[0, 0].set_title("Safety")

        axes[0, 1].plot(x_range, self.gap_log)
        axes[0, 1].set_ylabel("Dual gap")
        axes[0, 1].set_title("Duality gap")

        axes[1, 0].plot(x_range, self.performance_log_R)
        axes[1, 0].set_ylabel("Returns vs baseline (dotted black line)")
        axes[1, 0].axhline(y=pib_R_true_performance, label="baseline for 1st obj", linestyle="--", c='black')
        axes[1, 0].set_title("Performance for 1st objective")
        axes[1, 0].legend()

        neg_costs = [-1. * i for i in self.performance_log_C]
        axes[1, 1].plot(x_range, neg_costs)
        axes[1, 1].set_ylabel("Returns vs baseline (dotted red line)")
        axes[1, 1].axhline(y=- pib_C_true_performance, label="baseline for 2nd obj", linestyle="--", c='red')
        axes[1, 1].set_title("Performance for 2nd objective")
        axes[1, 1].legend()

        plt.show()

