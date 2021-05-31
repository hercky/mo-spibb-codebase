import numpy as np
import cvxpy as cp

def generate_iterates(xinit, operator, termination_condition):
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
    """
    for iterate in generate_iterates(xinit, operator, termination_condition):
        pass
    return iterate


def bounded_successive_approximation(xinit, operator=lambda x: x,
                                     termination_condition=lambda xprev, x: False,
                                     max_limit=50):
    """
    Iterations are bounded bt the max_limit variable
    """
    count = 0
    for iterate in generate_iterates(xinit, operator, termination_condition):
        count += 1
        if count >= max_limit:
            break

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


# make policy evaluation (direct method)
def reward_direct_policy_evaluation(P: np.ndarray,
                             R: np.ndarray,
                             discount: float,
                             policy: np.ndarray
                             ) -> np.ndarray:
    """
    Does policy evaluation by solving the system of equation
    instead of taking inverse
    (I - \gamma P^{\pi})^{-1}

    :param P: transition matrix (sat)
    :param R: reward matrix (sat)
    :param discount: discount factor
    :param policy:
    :return: vf: value function estimated
    """
    ppi = np.einsum('sat,sa->st', P, policy)
    rpi = np.einsum('sat,sat,sa->s', P, R, policy)
    vf = np.linalg.solve(np.eye(P.shape[-1]) - discount*ppi, rpi)
    return vf

def cost_direct_policy_evaluation(P: np.ndarray,
                             C: np.ndarray,
                             discount: float,
                             policy: np.ndarray
                             ) -> np.ndarray:
    """
    Does policy evaluation by solving the system of equation
    instead of taking inverse
    (I - \gamma P^{\pi})^{-1}

    :param P: transition matrix (sat)
    :param C: cost matrix (sa)
    :param discount: discount factor
    :param policy:
    :return: vf: value function estimated
    """
    ppi = np.einsum('sat,sa->st', P, policy)
    cpi = np.einsum('sa,sa->s', C, policy)
    vf = np.linalg.solve(np.eye(P.shape[-1]) - discount*ppi, cpi)
    return vf


# Policy Iteration operator (just uses R)
def make_policy_iteration_operator(P, R, discount, **kwargs):
    """
    P: sat
    R sat
    returns a operator that does 1 step of policy improvement
    via the Linear Programming formulation of the PI

    :param P: transition matrix
    """
    nstates = R.shape[0]
    nactions = R.shape[1]

    # convert to expected \E_{s'}[R(s,a,s')]
    R_sa = np.einsum('sat,sat -> sa', R, P)

    def lp_policy_iteration_operator(policy):
        # calculate the q-values
        v = reward_direct_policy_evaluation(P, R, discount, policy)
        Q = R_sa + discount * np.einsum('sat,t -> sa', P, v)

        # final policy placeholder
        soln_pi = np.zeros((nstates, nactions))

        # state-wise PI
        for s in range(nstates):

            # construct local pi
            pi = cp.Variable(shape=(1,nactions))
            obj = cp.Maximize(cp.sum(cp.multiply(pi, Q[[s]])))  # <Q(s,.), \pi(.|s)>

            # add lower bound constraint
            constr = [ pi[0] >= 0 ]

            # define the probability constraints
            constr += [ cp.sum(pi[0]) == 1 ]

            # solve
            prob = cp.Problem(obj, constr)
            prob.solve()

            # normalize the policy policy (if needed)
            new_policy = pi.value
            #new_policy = new_policy / new_policy.sum(axis=1)[:, None]

            # copy the solution for this state
            soln_pi[s] = new_policy[0]

        return soln_pi


    return lp_policy_iteration_operator



def compute_error_function(obs_counts, nstates: int, nactions: int, delta=1.0):
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

    # copy the counts
    for state in range(nstates):
        for action in range(nactions):
            count_sa[state,action] = sum(obs_counts[state, action, :])

    # for each (s,a)
    for s in range(nstates):
        for a in range(nactions):
            if count_sa[s, a] == 0.0:
                eQ[s, a] = np.inf
            else:
                eQ[s, a] = np.sqrt(2 * np.log(2 * ((nstates * nactions) / delta)) / count_sa[s, a])

    return eQ


def make_constrained_spibb_policy_iteration_operator(P, R, C, discount, baseline, error_fn, epsilon, coeffs, **kwargs):
    """
    IMP:

    coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0

    epsilon: if eps = np.inf, then (\pi_b,e_Q,eps) constraint is not enforced
    """
    nstates = R.shape[0]
    nactions = R.shape[1]

    # convert to expected \E_{s'}[R(s,a,s')], sat -> sa
    R_sa = np.einsum('sat,sat -> sa', R, P)

    # calculate the estimates for the baseline policy
    vR_b = reward_direct_policy_evaluation(P, R, discount, baseline)
    QR_b = R_sa + discount * np.einsum('sat,t -> sa', P, vR_b)
    AR_b = QR_b - vR_b.reshape((nstates, 1))

    vC_b = cost_direct_policy_evaluation(P, C, discount, baseline)
    QC_b = C + discount * np.einsum('sat,t -> sa', P, vC_b)
    AC_b = QC_b - vC_b.reshape((nstates, 1))

    def constrained_spibb_policy_iteration_operator(policy):

        # compute Q using direct policy evaluation
        # for the reward
        vR = reward_direct_policy_evaluation(P, R, discount, policy)
        QR = R_sa + discount * np.einsum('sat,t -> sa', P, vR)

        # for the cost
        vC = cost_direct_policy_evaluation(P, C, discount, policy)
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

            # NOTE: Add another constraint based on correction index that preserves the
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



def make_q_learning_operator(P, R, C, discount, coeffs):
    """
    P: sat
    R: sat
    C: sa
    
    """
    nstates = R.shape[0]
    nactions = R.shape[1]
    
    # convert to expected \E_{s'}[R(s,a,s')]
    R_sa = np.einsum('sat,sat -> sa', R, P)

    def q_learning_operator(policy):
        # calculate the q-values
        vR = reward_direct_policy_evaluation(P, R, discount, policy)
        QR = R_sa + discount * np.einsum('sat,t -> sa', P, vR)

        # for the cost
        vC = cost_direct_policy_evaluation(P, C, discount, policy)
        QC = C + discount * np.einsum('sat,t -> sa', P, vC)

        # create the objective
        QL = coeffs[0] * QR - coeffs[1] * QC
        
        # final policy placeholder
        soln_pi = np.zeros((nstates, nactions))
        
        # greedy maximization of policy
        soln_pi[np.arange(nstates), np.argmax(QL,axis=1)] = 1.0
    
        return soln_pi

    return q_learning_operator



def make_adv_q_learning(P, R, C, discount, baseline, coeffs):
    """
    IMP:

    coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0

    epsilon: if eps = np.inf, then (\pi_b,e_Q,eps) constraint is not enforced
    """
    nstates = R.shape[0]
    nactions = R.shape[1]

    # convert to expected \E_{s'}[R(s,a,s')], sat -> sa
    R_sa = np.einsum('sat,sat -> sa', R, P)

    # calculate the estimates for the baseline policy
    vR_b = reward_direct_policy_evaluation(P, R, discount, baseline)
    QR_b = R_sa + discount * np.einsum('sat,t -> sa', P, vR_b)
    AR_b = QR_b - vR_b.reshape((nstates, 1))

    vC_b = cost_direct_policy_evaluation(P, C, discount, baseline)
    QC_b = C + discount * np.einsum('sat,t -> sa', P, vC_b)
    AC_b = QC_b - vC_b.reshape((nstates, 1))

    def adv_q_learning_operator(policy):

        # compute Q using direct policy evaluation
        # for the reward
        vR = reward_direct_policy_evaluation(P, R, discount, policy)
        QR = R_sa + discount * np.einsum('sat,t -> sa', P, vR)

        # for the cost
        vC = cost_direct_policy_evaluation(P, C, discount, policy)
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



def make_reward_shaping_policy_iteration_operator(P, R, C, discount, coeffs):
    """
    IMP:

    coeffs: the list of coefficients (\lambda_i) for each signal AND \lamba_i >= 0
    """
    nstates = R.shape[0]
    nactions = R.shape[1]

    # convert to expected \E_{s'}[R(s,a,s')], sat -> sa
    R_sa = np.einsum('sat,sat -> sa', R, P)

    def reward_shaping_pi_operator(policy):
        # compute Q using direct policy evaluation
        vR = reward_direct_policy_evaluation(P, R, discount, policy)
        QR = R_sa + discount * np.einsum('sat,t -> sa', P, vR)

        # for the cost
        vC = cost_direct_policy_evaluation(P, C, discount, policy)
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
