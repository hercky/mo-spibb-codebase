import numpy as np
import cvxpy as cp
from typing import Callable, Generator, Tuple

def direct_policy_evaluation(P: np.ndarray,
                             R: np.ndarray,
                             discount: float,
                             policy: np.ndarray
                             ) -> np.ndarray:
    """
    Does policy evaluation by solving the system of equation
    instead of taking inverse
    (I - \gamma P^{\pi})^{-1}

    :param P: transition matrix
    :param R: reward matrix
    :param discount: discount factor
    :param policy:
    :return: vf: value function estimated
    """
    ppi = np.einsum('sat,sa->st', P, policy)
    rpi = np.einsum('sa,sa->s', R, policy)
    vf = np.linalg.solve(np.eye(P.shape[-1]) - discount*ppi, rpi)
    return vf


def generate_iterates(xinit: np.ndarray,
                      operator: Callable[[np.ndarray], np.ndarray],
                      termination_condition: Callable[[np.ndarray, np.ndarray], bool]
                      ) -> Generator[np.ndarray, None, None]:
    """

    :param xinit: initial value
    :param operator: operator to apply on the function
    :param termination_condition: checks when to terminate
    :return: a generator that gives the next iterates
    """
    x, xprev = operator(xinit), xinit
    yield x
    while not termination_condition(xprev, x):
        x, xprev = operator(x), x
        yield x


def successive_approximation(xinit: np.ndarray,
                             operator=lambda x: x,
                             termination_condition=lambda xprev, x: False):
    """
    Iteratively applies the operator until satisfied by the terminatation condition

    :param xinit: initial value
    :param operator: operator to apply on the function
    :param termination_condition: checks when to terminate
    :return:
    """
    for iterate in generate_iterates(xinit, operator, termination_condition):
        pass
    return iterate


def bounded_successive_approximation(xinit,
                                     operator=lambda x: x,
                                     termination_condition=lambda xprev, x: False,
                                     max_limit=50):
    """
    Iterations are bounded bt the max_limit variable

    :param xinit:
    :param operator:
    :param termination_condition:
    :param max_limit:
    :return:
    """
    count = 0
    for iterate in generate_iterates(xinit, operator, termination_condition):
        count += 1
        if count >= max_limit:
            break
        else:
            pass

    return iterate

def default_termination(xprev, x, epsilon=1e-8):
    """
    A standard termination condition
    :param xprev:
    :param x:
    :param epsilon:
    :return:
    """
    return np.linalg.norm(xprev - x) < epsilon


def cmdp_dual_lp(P: np.ndarray,
                 R: np.ndarray,
                 C: np.ndarray,
                 discount: float,
                 d0: float,
                 initial_distribution: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]  :
    """
    Solves the CMDP using the procedure described in Appendix D

    :param P: transition matrix of shape [|S|,|A|,|S|]
    :param R: shape [|S|,|A|]
    :param C: shape [|S|,|A|]
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
    constr += [cp.sum(cp.multiply(p, C)) <= d0]

    # sovle
    prob = cp.Problem(obj, constr)
    # prob.solve(verbose=False, solver=cp.CVXOPT)
    # Let CVXPY chose the best solver for this problem
    prob.solve(verbose=False)

    pi_opt = p.value
    pi_opt = pi_opt / pi_opt.sum(axis=1)[:, None]

    cum_cost = np.sum(np.asarray(p.value) * C)

    return prob.value, cum_cost, pi_opt


def generate_dataset(num_trajs: int, env, pi):
    """
    Generates a dataset of num_trajs for the env and a given policy pi

    Each tuple contains: state, action, reward, cost, next_state

    :param num_trajs: the size of dataset
    :param env:
    :param pi:
    :return: Tuple[trajectories, batch of transition]
    """
    trajectories = []
    for _ in range(num_trajs):
        traj = []
        # sample a single traj here
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(pi.shape[1], p=pi[state])
            next_state, reward, done, info = env.step(action)
            cost = info['pit']
            traj.append([state, action, reward, cost, next_state])
            state = next_state
        trajectories.append(traj)

    # unpack the individual transitions
    batch_transition = [val for sublist in trajectories for val in sublist]

    return trajectories, batch_transition


def estimate_model(batch, nstates, nactions, zero_unseen=True):
    """
    build the MLE transition \hat{P} here

    :param batch:
    :param nstates:
    :param nactions:
    :param zero_unseen: a flag to take handle for unseen transitions
    :return:
    """
    count_P = np.zeros((nstates, nactions, nstates))
    for transition in batch:
        state = transition[0]
        action = transition[1]
        next_state = transition[-1]

        count_P[state, action, next_state] += 1.0

    # do the normalization here
    est_P = count_P / np.sum(count_P, 2)[:, :, np.newaxis]

    if zero_unseen:
        est_P = np.nan_to_num(est_P)
    else:
        est_P[np.isnan(est_P)] = 1.0/nstates

    return est_P


def compute_error_function(batch, nstates: int, nactions: int, delta=1.0):
    """
    Computes the e_Q function based on the dataset

    :param batch:
    :param nstates:
    :param nactions:
    :param delta:
    :return:
    """
    count_sa = np.zeros((nstates, nactions))
    eQ = np.zeros((nstates, nactions))

    # for each transition in batch
    for transition in batch:
        state = transition[0]
        action = transition[1]

        count_sa[state, action] += 1.0

    # for each (s,a)
    for s in range(nstates):
        for a in range(nactions):
            if count_sa[s, a] == 0.0:
                eQ[s, a] = np.inf
            else:
                eQ[s, a] = np.sqrt(2 * np.log(2 * ((nstates * nactions) / delta)) / count_sa[s, a])

    return eQ