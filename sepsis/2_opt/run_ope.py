"""
Evaluation script for using OPE for evaluating the performance of different policies returned from run_pi
    and save the results in the dataframe
"""
import logging as log
import argparse
import os

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import time

from utils import reward_direct_policy_evaluation, cost_direct_policy_evaluation
from eval_utils import Evaluator

# ------------------------------------------------------------
#                  Set up parameters and variables
# ------------------------------------------------------------

# ----------- Defaults MDP parameters -------------
nS, nA = 750, 25
DEATH_STATE = 750
SURVIVAL_STATE = 751

gamma = 0.99

# ----------- Search space  -----------
LAMBDA_R_vals = [1.0, 0.0]
# LAMBDA_C_vals = [1.0, 0.0, 0.01]
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
    '--n_bootstrap', type=int, default=1000, help='the cost for taking rare action')

args = parser.parse_args()

SEED = args.seed
FREQUENCY_THRESHOLD = args.freq
COST_FOR_RARE_DECISION = args.cost

# -------- Folder Paths -------------
basepath = '/enter/path/here/'
# Path variables
LOG_PATH = f'{basepath}/logs'
IMPORT_PATH = f'{basepath}/m_hat/{SEED}'
OUTPUT_PATH = f'{basepath}/output/{SEED}/freq_{FREQUENCY_THRESHOLD}_cost_{COST_FOR_RARE_DECISION}'

# -------- Log -------------
log.basicConfig(
    filename=f'{LOG_PATH}/{SEED}-freq_{FREQUENCY_THRESHOLD}_cost_{COST_FOR_RARE_DECISION}-eval.log', filemode='w',
    format='%(asctime)s - %(levelname)s \t %(message)s',
    level=log.DEBUG)

# ---------    Main code starts here --------------
log.info("Starting the Eval script")

results = []

# ------------------------------------------------------------
#                  5_5-OPE
# ------------------------------------------------------------
log.info("----- 5-5 OPE ------")

log.info("Loading m_hat matrices")
P_mat, R_mat = pickle.load(open(f"{IMPORT_PATH}/MDP_mat.p", "rb"))

# load the defaults for this seed
C_mat = pickle.load(open(f"{OUTPUT_PATH}/C_mat.p", "rb"))
pi_baseline = pickle.load(open(f"{OUTPUT_PATH}/pi_baseline.p", "rb"))

# check if the same cost was used for both runs
assert COST_FOR_RARE_DECISION == np.max(C_mat)

# get R_sa once
R_sa = np.einsum('sat,sat -> sa', R_mat, P_mat)

# create the evaluator object
evaluator = Evaluator(gamma=gamma,
                      pi_baseline=pi_baseline,
                      C_mat=C_mat,
                      cost_for_rare_decision=COST_FOR_RARE_DECISION,
                      n_bootstrap=args.n_bootstrap, )

# ------------------------------------------------------------
#          Calculate validation and test sets stats
# ------------------------------------------------------------

# --- Test set ---
log.info("Loading test set")
traj_te = pickle.load(open(f'{IMPORT_PATH}/trajDr_te.pkl', 'rb'))

test_trajectories = evaluator.preprocess_trajecteories(traj_te)
N_test = len(test_trajectories)
log.info(f'Effective sample size of test set {N_test}')

# get the mean
test_mean_stats = evaluator.get_mean_stats(test_trajectories)
log.info(f'--- Mean stats --- \n R: {test_mean_stats[0]}, C: {test_mean_stats[1]}')

r_test_stats, c_test_stats = evaluator.get_bootstrap_stats(test_trajectories)
log.info(f'-- Bootstrap stats (n_bootstap = {evaluator.n_bootstrap}) ---')
log.info(f'R: {r_test_stats[0]} +/- {r_test_stats[1]}')
log.info(f'C: {c_test_stats[0]} +/- {c_test_stats[1]}')

# add to the results
results.append([SEED, 'baseline',
                None, None,  # lambda is not valid for baseline
                'test',  # train/tes/valid
                test_mean_stats[0], test_mean_stats[1],  # append the mean R and C
                r_test_stats[0], r_test_stats[1],  # R_boot stats
                c_test_stats[0], c_test_stats[1],  # C_boot stats
                0.0,  # pi_diff is 0 as this is baseline
                None,  # eps None
                None, None,  # delta not valid
                None,  # OPE not valid
                None,  # Lower bound not valid
                ])

# --- Validation set ---
log.info("Loading validation set")
traj_va = pickle.load(open(f'{IMPORT_PATH}/trajDr_va.pkl', 'rb'))

va_trajectories = evaluator.preprocess_trajecteories(traj_va)
log.info(f'Effective sample size of validation set {len(va_trajectories)}')

# get the mean
va_mean_stats = evaluator.get_mean_stats(va_trajectories)
log.info(f'--- Mean stats --- \n R: {va_mean_stats[0]}, C: {va_mean_stats[1]}')

# get the bootsrap estimates here
r_va_stats, c_va_stats = evaluator.get_bootstrap_stats(va_trajectories)
log.info(f'-- Bootstrap stats (n_bootstap = {evaluator.n_bootstrap}) ---')
log.info(f'R: {r_va_stats[0]} +/- {r_va_stats[1]}')
log.info(f'C: {c_va_stats[0]} +/- {c_va_stats[1]}')

# add to the results
results.append([SEED, 'baseline',
                None, None,  # lambda is not valid for baseline
                'validation',  # train/tes/valid
                va_mean_stats[0], va_mean_stats[1],  # append the mean R and C
                r_va_stats[0], r_va_stats[1],  # R_boot stats
                c_va_stats[0], c_va_stats[1],  # C_boot stats
                0.0,  # pi_diff is 0 as this is baseline
                None,  # eps None
                None, None,  # delta not valid
                None,  # OPE not valid
                None,  # Lower bound not valid
                ])



# ------------------------------------------------------------
#          Helper function for OPE estimation
# ------------------------------------------------------------

def ope_estimation(pi_solution, alg_name, lR, lC, eps):
    """
    helper function to do DR estimation

    does both for test and validation sets


    :param alg_name: 'regular_PI'/
    :param pi_solution:
    :return:
    """
    # calculate the change in pi (L2-norm)
    pi_diff = np.linalg.norm(pi_solution - pi_baseline)

    # estimate the value function estimates needed for OPE
    # R
    rV_sol = reward_direct_policy_evaluation(P_mat, R_mat, gamma, pi_solution)
    rQ_sol = R_sa + gamma * np.einsum('sat,t -> sa', P_mat, rV_sol)
    # C
    cV_sol = cost_direct_policy_evaluation(P_mat, C_mat, gamma, pi_solution)
    cQ_sol = C_mat + gamma * np.einsum('sat,t -> sa', P_mat, cV_sol)

    # ---------------------------------------------
    #                    Test
    # ---------------------------------------------

    # ------- Doubly Robust OPE ---------------
    dr_mean_stats, dr_r_stats_boot, dr_c_stats_boot = evaluator.doubly_robust_ope(test_trajectories,
                                                                                  pi_e=pi_solution,
                                                                                  rQ_e=rQ_sol,
                                                                                  cQ_e=cQ_sol,
                                                                                  )

    log.info('-- (TEST) Mean stats ---')
    log.info('-- Doubly Robust (DR) ---')
    log.info(f'R: {dr_mean_stats[0]}, C: {dr_mean_stats[1]}')
    log.info(f'-- Bootstrap stats (n_bootstap = {evaluator.n_bootstrap}) ---')
    log.info(f'R: {dr_r_stats_boot[0]} +/- {dr_r_stats_boot[1]}')
    log.info(f'C: {dr_c_stats_boot[0]} +/- {dr_c_stats_boot[1]}')

    # save to the results list
    results.append([SEED, alg_name,  # seed, alg_name
                    lR, lC,  # 'lambda_R', 'lambda_C',
                    'test',  # 'dataset',#train/tes/valid
                    dr_mean_stats[0], dr_mean_stats[1],  # 'R_mean', 'C_mean',
                    dr_r_stats_boot[0], dr_r_stats_boot[1],  # 'R_boot_mean', 'R_boot_std',
                    dr_c_stats_boot[0], dr_c_stats_boot[1],  # 'C_boot_mean', 'C_boot_std',
                    pi_diff,
                    eps,
                    None, None,  # 'delta', 'delta_hcpi',
                    'DR',  # 'ope-estimator',
                    'bootstrap',  # lower_bound_strategy
                    ])

    # ------- Weighted Doubly Robust OPE ---------------
    dr_mean_stats, dr_r_stats_boot, dr_c_stats_boot = evaluator.weighted_doubly_robust_ope(test_trajectories,
                                                                                  pi_e=pi_solution,
                                                                                  rQ_e=rQ_sol,
                                                                                  cQ_e=cQ_sol,
                                                                                  )

    log.info('-- Weighted Doubly Robust (WDR) ---')
    log.info(f'R: {dr_mean_stats[0]}, C: {dr_mean_stats[1]}')
    log.info(f'-- Bootstrap stats (n_bootstap = {evaluator.n_bootstrap}) ---')
    log.info(f'R: {dr_r_stats_boot[0]} +/- {dr_r_stats_boot[1]}')
    log.info(f'C: {dr_c_stats_boot[0]} +/- {dr_c_stats_boot[1]}')

    # save to the results list
    results.append([SEED, alg_name,  # seed, alg_name
                    lR, lC,  # 'lambda_R', 'lambda_C',
                    'test',  # 'dataset',#train/tes/valid
                    dr_mean_stats[0], dr_mean_stats[1],  # 'R_mean', 'C_mean',
                    dr_r_stats_boot[0], dr_r_stats_boot[1],  # 'R_boot_mean', 'R_boot_std',
                    dr_c_stats_boot[0], dr_c_stats_boot[1],  # 'C_boot_mean', 'C_boot_std',
                    pi_diff,
                    eps,
                    None, None,  # 'delta', 'delta_hcpi',
                    'WDR',  # 'ope-estimator',
                    'bootstrap',  # lower_bound_strategy
                    ])

    # ---------------------------------------------
    #                    Validation
    # ---------------------------------------------

    # ------- Doubly Robust OPE ---------------
    dr_mean_stats, dr_r_stats_boot, dr_c_stats_boot = evaluator.doubly_robust_ope(va_trajectories,
                                                                                  pi_e=pi_solution,
                                                                                  rQ_e=rQ_sol,
                                                                                  cQ_e=cQ_sol,
                                                                                  )

    log.info('-- (VALIDATION) Mean stats ---')
    log.info('-- Doubly Robust (DR) ---')
    log.info(f'R: {dr_mean_stats[0]}, C: {dr_mean_stats[1]}')
    log.info(f'-- Bootstrap stats (n_bootstap = {evaluator.n_bootstrap}) ---')
    log.info(f'R: {dr_r_stats_boot[0]} +/- {dr_r_stats_boot[1]}')
    log.info(f'C: {dr_c_stats_boot[0]} +/- {dr_c_stats_boot[1]}')

    # save to the results list
    results.append([SEED, alg_name,  # seed, alg_name
                    lR, lC,  # 'lambda_R', 'lambda_C',
                    'validation',  # 'dataset',#train/tes/valid
                    dr_mean_stats[0], dr_mean_stats[1],  # 'R_mean', 'C_mean',
                    dr_r_stats_boot[0], dr_r_stats_boot[1],  # 'R_boot_mean', 'R_boot_std',
                    dr_c_stats_boot[0], dr_c_stats_boot[1],  # 'C_boot_mean', 'C_boot_std',
                    pi_diff,
                    eps,
                    None, None,  # 'delta', 'delta_hcpi',
                    'DR',  # 'ope-estimator',
                    'bootstrap',  # lower_bound_strategy
                    ])

    # ------- Weighted Doubly Robust OPE ---------------
    dr_mean_stats, dr_r_stats_boot, dr_c_stats_boot = evaluator.weighted_doubly_robust_ope(va_trajectories,
                                                                                           pi_e=pi_solution,
                                                                                           rQ_e=rQ_sol,
                                                                                           cQ_e=cQ_sol,
                                                                                           )

    log.info('-- Weighted Doubly Robust (WDR) ---')
    log.info(f'R: {dr_mean_stats[0]}, C: {dr_mean_stats[1]}')
    log.info(f'-- Bootstrap stats (n_bootstap = {evaluator.n_bootstrap}) ---')
    log.info(f'R: {dr_r_stats_boot[0]} +/- {dr_r_stats_boot[1]}')
    log.info(f'C: {dr_c_stats_boot[0]} +/- {dr_c_stats_boot[1]}')

    # save to the results list
    results.append([SEED, alg_name,  # seed, alg_name
                    lR, lC,  # 'lambda_R', 'lambda_C',
                    'validation',  # 'dataset',#train/tes/valid
                    dr_mean_stats[0], dr_mean_stats[1],  # 'R_mean', 'C_mean',
                    dr_r_stats_boot[0], dr_r_stats_boot[1],  # 'R_boot_mean', 'R_boot_std',
                    dr_c_stats_boot[0], dr_c_stats_boot[1],  # 'C_boot_mean', 'C_boot_std',
                    pi_diff,
                    eps,
                    None, None,  # 'delta', 'delta_hcpi',
                    'WDR',  # 'ope-estimator',
                    'bootstrap',  # lower_bound_strategy
                    ])

# ------------------------------------------------------------
#               Eval regular PI solutions
# ------------------------------------------------------------

log.info(f'Evaluating regular PI solution')

# load a solution policy
pi_solution = pickle.load(open(f'{OUTPUT_PATH}/regular_PI_solution.p', 'rb'))

# does the ope estimation heavy-lifting
ope_estimation(pi_solution=pi_solution,alg_name= 'regular_PI', lR=None, lC=None, eps=None)


# ------------------------------------------------------------
#                  Eval baseline solutions (eps=0)
# ------------------------------------------------------------

log.info(f'Evaluating baseline solution')

# copy the solution to be baseline policy
# to get an estimate of how reliable IS estimators are
pi_solution = np.copy(pi_baseline)

ope_estimation(pi_solution=pi_solution,alg_name= 'baseline_copy', lR=None, lC=None, eps=None)


# -----------------------------------------------------------------------
#        Eval different (lambda_R-lambda_C and epsilon) solutions now
# -----------------------------------------------------------------------

for coeffs in LAMBDA_COEFFS:

    # Evaluate and store the corresponding reward-shaping solutions here
    log.info(f'Evaluating Reward Shaping with lambda_R={coeffs[0]}, lambda_C={coeffs[1]}')

    # load a solution policy
    sol_name = f'rs_{coeffs[0]}_{coeffs[1]}.p'
    pi_solution = pickle.load(open(f'{OUTPUT_PATH}/{sol_name}', 'rb'))
    # does the ope estimation heavy-lifting
    ope_estimation(pi_solution=pi_solution, alg_name='reward_shaping', lR=coeffs[0], lC=coeffs[1], eps=None)

    # ------------------------------------------------------------
    #                  Eval different \epsilons here
    # ------------------------------------------------------------
    for eps in EPS_LIST:
        log.info(f'Evaluating lambda_R={coeffs[0]}, lambda_C={coeffs[1]} with epsilon={eps}')

        # load a solution policy
        sol_name = f'cpi_{coeffs[0]}_{coeffs[1]}_{eps}.p'

        pi_solution = pickle.load(open(f'{OUTPUT_PATH}/{sol_name}', 'rb'))

        # does the ope estimation heavy-lifting
        ope_estimation(pi_solution=pi_solution, alg_name='sopt', lR=coeffs[0], lC=coeffs[1], eps=eps)


# -----------------------------------------------------------------------
#        Save everything as pandas df
# -----------------------------------------------------------------------

# save the results as a dataframe
df = pd.DataFrame(results, columns=['seed', 'algorithm',
                                    'lambda_R', 'lambda_C',
                                    'dataset',  # train/tes/valid
                                    'R_mean', 'C_mean',
                                    'R_boot_mean', 'R_boot_std', 'C_boot_mean', 'C_boot_std',
                                    'pi_diff',
                                    'epsilon',
                                    'delta', 'delta_hcpi',
                                    'ope-estimator',
                                    'lower_bound_strategy',
                                    ])

df.to_csv(path_or_buf=f'{OUTPUT_PATH}/results.csv')
df.to_pickle(path=f'{OUTPUT_PATH}/results.pkl')

# ------- Done ----------------
log.info("Done")
