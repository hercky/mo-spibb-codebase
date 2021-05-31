"""
Script for running the different Policy Iteration agents: Linearized, Adv-Linearized, S-OPT
and save the policies in outputs

"""
import logging as log
import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm

import cvxpy as cp
import time

from utils import *

# ------------------------------------------------------------
#                  Set up parameters and variables
# ------------------------------------------------------------

# ----------- Defaults MDP parameters -------------
nS, nA = 750, 25
DEATH_STATE = 750
SURVIVAL_STATE = 751

gamma = 0.99
theta = 1e-10

# ----------- Search space  -----------
LAMBDA_R_vals = [1.0, 0.0]
LAMBDA_C_vals = [1.0, 0.0, 0.1, 0.01,]
LAMBDA_COEFFS = [(lr, lc) for lc in LAMBDA_C_vals for lr in LAMBDA_R_vals]
EPS_LIST = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, np.inf]

# -------- User inputs -------------
parser = argparse.ArgumentParser(description="For running the PI methods with constraints")

parser.add_argument(
    '--seed', type=int, default=0, help='the seed that defines the input data to work with ')
parser.add_argument(
    '--freq', type=float, default=10.0, help='the frequency that determines what counts as a rare action')
parser.add_argument(
    '--cost', type=float, default=10.0, help='the cost for taking rare action')
parser.add_argument(
    '--n_iters', type=int, default=10, help='the max number of iterations to run the PI procedure for')

args = parser.parse_args()

SEED = args.seed
FREQUENCY_THRESHOLD = args.freq
COST_FOR_RARE_DECISION = args.cost
MAX_ITERS = args.n_iters

# -------- Folder Paths -------------
basepath = '/enter/path/here/'
# Path variables
LOG_PATH = f'{basepath}/logs'
IMPORT_PATH = f'{basepath}/m_hat/{SEED}'
EXPORT_PATH = f'{basepath}/output/{SEED}/freq_{FREQUENCY_THRESHOLD}_cost_{COST_FOR_RARE_DECISION}'

if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

# -------- Log -------------
log.basicConfig(
        filename=f'{LOG_PATH}/{SEED}-freq_{FREQUENCY_THRESHOLD}_cost_{COST_FOR_RARE_DECISION}-cpi.log', filemode='w',
        format='%(asctime)s - %(levelname)s \t %(message)s',
        level=log.DEBUG)

# ---------    Main code starts here --------------
log.info("Starting the PI algorithms scripts")

# ------------------------------------------------------------
#                  5_3-PI
# ------------------------------------------------------------
log.info("----- 5-3 PI ------")

log.info("Loading m_hat matrices")
P_mat, R_mat = pickle.load(open(f"{IMPORT_PATH}/MDP_mat.p", "rb"))
counts_mat = pickle.load(open(f"{IMPORT_PATH}/MDP_counts.p", "rb"))

# compute the (s,a) counts
count_sa = counts_mat.sum(axis=-1)

# add uniform policy to pi for the absorbing states!
random_policy = np.ones((nS+2, nA))/nA


# create the PI operator
pi_operator = make_policy_iteration_operator(P=P_mat, R=R_mat, discount=gamma)

start_time = time.time()
reg_pi_solution = bounded_successive_approximation(random_policy,
                                                   operator=pi_operator,
                                                   termination_condition=default_termination,
                                                   max_limit= MAX_ITERS,)

time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
log.info(f"Regular PI completed in {time_elapsed}")

# save the solution here
with open(f'{EXPORT_PATH}/regular_PI_solution.p', 'wb') as f:
    pickle.dump(reg_pi_solution, f)


# ------------------------------------------------------------
#           5_4-C_PI (Linearized/Adv-Linearized/S-OPT)
# ------------------------------------------------------------
log.info("----- 5-4 C-PI ------")

log.info("Creating C matrix")
# cost is state-action dependent
C_mat = np.zeros((nS+2, nA))

# Fill the Cost_mat here
for s in tqdm(range(nS)):
    for a in range(nA):
        C_mat[s, a] = (count_sa[s][a] < FREQUENCY_THRESHOLD) * COST_FOR_RARE_DECISION

log.info("Compute e_Q")
error_Q = compute_error_function(counts_mat, nS+2, nA)

log.info("Load the clinician's policy (baseline policy)")
pi_0_mat = pickle.load(open(f'{IMPORT_PATH}/clinician_policy_mat.p', 'rb'))

# append the random policy for absorbing states
pi_baseline = np.vstack([pi_0_mat, random_policy[-2:]])

# save the policies
with open(f'{EXPORT_PATH}/pi_baseline.p', 'wb') as f:
    pickle.dump(pi_baseline, f)

# C_mat
with open(f'{EXPORT_PATH}/C_mat.p', 'wb') as f:
    pickle.dump(C_mat, f)

for coeffs in LAMBDA_COEFFS:
    # for each coeff combination do reward shaping baseline first

    log.info(f"Starting the reward shaping for coeffs={coeffs}")
    reward_shpaing_operator = make_reward_shaping_policy_iteration_operator(P=P_mat, R=R_mat, C=C_mat,
                                                                            discount=gamma,
                                                                            coeffs=coeffs,
                                                                            )
    start_time = time.time()
    pi_solution = bounded_successive_approximation(random_policy,
                                                   operator=reward_shpaing_operator,
                                                   termination_condition=default_termination,
                                                   max_limit=MAX_ITERS, )

    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    log.info(f"Reward Shaping-PI completed in {time_elapsed}")

    # save the solution to a file here
    sol_name = f'rs_{coeffs[0]}_{coeffs[1]}.p'
    with open(f'{EXPORT_PATH}/{sol_name}', 'wb') as f:
        pickle.dump(pi_solution, f)

    # now try different eps for S-OPT
    for eps in EPS_LIST:
        # do PI for each combination
        log.info(f"Starting the C-PI for coeffs={coeffs} and epsilon={eps}")

        constrained_spibb_operator = make_constrained_spibb_policy_iteration_operator(P=P_mat, R=R_mat, C=C_mat,
                                                                                      discount=gamma,
                                                                                      baseline=pi_baseline,
                                                                                      error_fn=error_Q,
                                                                                      coeffs=coeffs,
                                                                                      epsilon=eps,
                                                                                      )

        start_time = time.time()
        pi_solution = bounded_successive_approximation(random_policy,
                                                       operator=constrained_spibb_operator,
                                                       termination_condition=default_termination,
                                                       max_limit=MAX_ITERS, )

        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        log.info(f"C-PI completed in {time_elapsed}")

        # save the solution to a file here
        sol_name = f'cpi_{coeffs[0]}_{coeffs[1]}_{eps}.p'

        with open(f'{EXPORT_PATH}/{sol_name}', 'wb') as f:
            pickle.dump(pi_solution, f)


# ------- Done ----------------
log.info("Done")
