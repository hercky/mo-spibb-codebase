from tqdm import tqdm
from sklearn import utils
import numpy as np


class Evaluator:
    """
    Class that evaluates the performae
    """

    def __init__(self, gamma, C_mat, pi_baseline, cost_for_rare_decision, n_bootstrap=1000):
        """

        :param gamma:
        :param C_mat:
        :param pi_baseline:
        :param n_bootstrap:
        """
        self.gamma = gamma
        self.C_mat = np.copy(C_mat)
        self.pi_baseline = pi_baseline
        self.n_bootstrap = n_bootstrap
        self.cost_for_rare_decision = cost_for_rare_decision

    def preprocess_trajecteories(self, trajectories):
        """
        Does the following steps:
        - Add costs to the trajectories using C_mat
        - Filter out unusable trajectories

        """
        # add costs to the trajectories
        for traj in trajectories:
            for transition in traj:
                s = transition['s']
                a = transition['a']
                transition['c'] = self.C_mat[s, a]

        ok_trajectories = []

        # Filter out unusable trajectories
        # must not contain (s,a) pairs not observed in the training set
        test_trajectories = []
        for traj in trajectories:
            usable = True
            for transition in traj:
                s = transition['s']
                a = transition['a']
                if np.isclose(self.pi_baseline[s, a], 0.0):
                    usable = False
                    break
            if usable:
                ok_trajectories.append(traj)

        return ok_trajectories

    def get_mean_stats(self, trajectories):
        """
        Get the stats (mean,std_dev) using bootstrap for a set of trajectories
        """
        # Observed  returns
        rV_DATA = []
        cV_DATA = []

        for i, traj in enumerate(trajectories):
            H = len(traj)
            rG = 0
            cG = 0
            for t in reversed(range(H)):
                rG = traj[t]['r'] + self.gamma * rG
                cG = traj[t]['c'] + self.gamma * cG
            rV_DATA.append(rG)
            cV_DATA.append(cG)

        return np.mean(rV_DATA), np.mean(cV_DATA)

    def get_bootstrap_stats(self, trajectories):
        """
        Get the stats (mean,std_dev) using bootstrap for a set of trajectories
        """

        V_DATA = []
        for i in tqdm(range(self.n_bootstrap)):
            traj_boot = utils.resample(trajectories, replace=True, random_state=i)
            V_DATA.append(self.get_mean_stats(traj_boot))

        rV_DATA = [i[0] for i in V_DATA]
        cV_DATA = [i[1] for i in V_DATA]

        r_mean = np.mean(rV_DATA)
        r_std = np.std(rV_DATA)

        c_mean = np.mean(cV_DATA)
        c_std = np.std(cV_DATA)

        return (r_mean, r_std), (c_mean, c_std)

    def calculate_rho_cum(self, trajectories, pi_e):
        """
        Get the IS ratios matrix and the weights
        """
        # Calculate all per-step importance sampling ratio
        rho_all = []
        for traj in trajectories:
            rho = []
            for transition in traj:
                s = transition['s']
                a = transition['a']
                rho_t = pi_e[s, a] / self.pi_baseline[s, a]
                rho.append(rho_t)
            rho_all.append(np.array(rho))

        # Find out the maximum trajectory length
        max_H = max(len(traj) for traj in trajectories)

        # size of dataset
        N = len(trajectories)

        # Calculate cumulative importance ratio, rho_{1:t} for each trajectory at each timestep
        rho_cum = np.zeros((N, max_H))
        for i, rho in enumerate(rho_all):
            rho_tmp = np.ones(max_H)
            rho_tmp[:len(rho)] = rho
            rho_cum[i] = np.cumprod(rho_tmp)

        # Calculate the average cumulative importance ratio at every horizon t
        weights = rho_cum.mean(axis=0)

        return rho_cum, weights

    def doubly_robust_ope(self, trajectories, pi_e, rQ_e, cQ_e):
        """

        trajectories: the dataset to work on
        pi_sol: evaluation policy
        estimated Q function for pi_sol
        """
        # compute the estimated Q functions

        max_H = max(len(traj) for traj in trajectories)

        rho_cum, weights = self.calculate_rho_cum(trajectories, pi_e)

        V_DR = []

        # report the means instead
        for traj, rho_cumulative in zip(trajectories, rho_cum):
            V_DR.append(doubly_robust_estimator(trajectory=traj,
                                                rQ=rQ_e, cQ=cQ_e,
                                                pi_0=self.pi_baseline, pi_e=pi_e,
                                                rho_cumulative=rho_cumulative,
                                                gamma=self.gamma))

        # separate the reward and cost tuples into different lists
        rV_DR, cV_DR = zip(*V_DR)

        # clip the estimates in the appropriate range
        r_mean = np.mean(np.clip(rV_DR, -100, 100))
        c_mean = np.mean(np.clip(cV_DR, 0, max_H * self.cost_for_rare_decision))

        rV_DR_b = []
        cV_DR_b = []
        # return the bootstrapped estimates
        for i in tqdm(range(self.n_bootstrap)):
            V_DR_boot = utils.resample(V_DR, replace=True, random_state=i)
            rV_DR_boot, cV_DR_boot = zip(*V_DR_boot)

            # clip
            rV_DR_b.append(np.mean(np.clip(rV_DR_boot, -100, 100)))
            cV_DR_b.append(np.mean(np.clip(cV_DR_boot, 0, max_H * self.cost_for_rare_decision)))

        r_mean_b = np.mean(rV_DR_b)
        r_std_b = np.std(rV_DR_b)

        c_mean_b = np.mean(cV_DR_b)
        c_std_b = np.std(cV_DR_b)

        return (r_mean, c_mean), (r_mean_b, r_std_b), (c_mean_b, c_std_b)

    def weighted_doubly_robust_ope(self, trajectories, pi_e, rQ_e, cQ_e):
        """

        trajectories: the dataset to work on
        pi_sol: evaluation policy
        estimated Q function for pi_sol
        """
        # compute the estimated Q functions

        max_H = max(len(traj) for traj in trajectories)

        rho_cum, weights = self.calculate_rho_cum(trajectories, pi_e)

        V_DR = []

        # report the means instead
        for traj, rho_cumulative in zip(trajectories, rho_cum):
            V_DR.append(weighted_doubly_robust_estimator(trajectory=traj,
                                                         rQ=rQ_e, cQ=cQ_e,
                                                         pi_0=self.pi_baseline, pi_e=pi_e,
                                                         rho_cumulative=rho_cumulative,
                                                         weight_t=weights,
                                                         gamma=self.gamma))

        # separate the reward and cost tuples into different lists
        rV_DR, cV_DR = zip(*V_DR)

        # clip the estimates in the appropriate range
        r_mean = np.mean(np.clip(rV_DR, -100, 100))
        c_mean = np.mean(np.clip(cV_DR, 0, max_H * self.cost_for_rare_decision))

        rV_DR_b = []
        cV_DR_b = []
        # return the bootstrapped estimates
        for i in tqdm(range(self.n_bootstrap)):
            V_DR_boot = utils.resample(V_DR, replace=True, random_state=i)
            rV_DR_boot, cV_DR_boot = zip(*V_DR_boot)

            # clip
            rV_DR_b.append(np.mean(np.clip(rV_DR_boot, -100, 100)))
            cV_DR_b.append(np.mean(np.clip(cV_DR_boot, 0, max_H * self.cost_for_rare_decision)))

        r_mean_b = np.mean(rV_DR_b)
        r_std_b = np.std(rV_DR_b)

        c_mean_b = np.mean(cV_DR_b)
        c_std_b = np.std(cV_DR_b)

        return (r_mean, c_mean), (r_mean_b, r_std_b), (c_mean_b, c_std_b)


    def list_format_doubly_robust_ope(self, trajectories, pi_e, rQ_e, cQ_e):
        """

        trajectories: the dataset to work on
        pi_sol: evaluation policy
        estimated Q function for pi_sol
        """
        max_H = max(len(traj) for traj in trajectories)
        rho_cum, weights = self.calculate_rho_cum(trajectories, pi_e)

        V_DR = []
        for traj, rho_cumulative in zip(trajectories, rho_cum):
            V_DR.append(doubly_robust_estimator(trajectory=traj,
                                                rQ=rQ_e, cQ=cQ_e,
                                                pi_0=self.pi_baseline, pi_e=pi_e,
                                                rho_cumulative=rho_cumulative,
                                                gamma=self.gamma))

        # separate the reward and cost tuples into different lists
        rV_DR, cV_DR = zip(*V_DR)

        # clip the estimates in the appropriate range
        rV_DR = np.clip(rV_DR, -100, 100)
        cV_DR = np.clip(cV_DR, 0, max_H * self.cost_for_rare_decision)

        return rV_DR, cV_DR

    def list_format_weighted_doubly_robust_ope(self, trajectories, pi_e, rQ_e, cQ_e):
        """

        trajectories: the dataset to work on
        pi_sol: evaluation policy
        estimated Q function for pi_sol
        """
        # compute the estimated Q functions
        max_H = max(len(traj) for traj in trajectories)
        rho_cum, weights = self.calculate_rho_cum(trajectories, pi_e)

        V_DR = []
        for traj, rho_cumulative in zip(trajectories, rho_cum):
            V_DR.append(weighted_doubly_robust_estimator(trajectory=traj,
                                                         rQ=rQ_e, cQ=cQ_e,
                                                         pi_0=self.pi_baseline, pi_e=pi_e,
                                                         rho_cumulative=rho_cumulative,
                                                         weight_t=weights,
                                                         gamma=self.gamma))

        # separate the reward and cost tuples into different lists
        rV_DR, cV_DR = zip(*V_DR)

        # clip the estimates in the appropriate range
        rV_DR = np.clip(rV_DR, -100, 100)
        cV_DR = np.clip(cV_DR, 0, max_H * self.cost_for_rare_decision)

        return rV_DR, cV_DR

def doubly_robust_estimator(trajectory, rQ, cQ, pi_0, pi_e, rho_cumulative, gamma):
    rV_DR = 0
    cV_DR = 0
    T = len(trajectory)
    for t in range(T):
        transition = trajectory[t]
        s = transition['s']
        a = transition['a']
        r = transition['r']
        c = transition['c']

        rQ_hat = rQ[s, a]
        rV_hat = np.nansum(rQ[s] * pi_e[s])
        cQ_hat = cQ[s, a]
        cV_hat = np.nansum(cQ[s] * pi_e[s])
        assert not np.isclose(pi_0[s, a], 0.0)

        rho_1t = rho_cumulative[t]
        if t == 0:
            rho_1t_1 = 1.0
        else:
            rho_1t_1 = rho_cumulative[t - 1]

        rV_DR = rV_DR + np.power(gamma, t) * (rho_1t * r - (rho_1t * rQ_hat - rho_1t_1 * rV_hat))
        cV_DR = cV_DR + np.power(gamma, t) * (rho_1t * c - (rho_1t * cQ_hat - rho_1t_1 * cV_hat))

    return rV_DR, cV_DR


def weighted_doubly_robust_estimator(trajectory, rQ, cQ, pi_0, pi_e, rho_cumulative, weight_t, gamma):
    rV_WDR = 0
    cV_WDR = 0
    T = len(trajectory)
    for t in range(T):
        transition = trajectory[t]
        s = transition['s']
        a = transition['a']
        r = transition['r']
        c = transition['c']

        rQ_hat = rQ[s, a]
        rV_hat = np.nansum(rQ[s] * pi_e[s])
        cQ_hat = cQ[s, a]
        cV_hat = np.nansum(cQ[s] * pi_e[s])
        assert not np.isclose(pi_0[s, a], 0.0)

        rho_1t = rho_cumulative[t] / weight_t[t]
        if t == 0:
            rho_1t_1 = 1.0
        else:
            rho_1t_1 = rho_cumulative[t - 1] / weight_t[t - 1]

        rV_WDR = rV_WDR + np.power(gamma, t) * (rho_1t * r - (rho_1t * rQ_hat - rho_1t_1 * rV_hat))
        cV_WDR = cV_WDR + np.power(gamma, t) * (rho_1t * c - (rho_1t * cQ_hat - rho_1t_1 * cV_hat))

    return rV_WDR, cV_WDR

