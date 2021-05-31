"""
Evaluation script for using OPE in notebook in 5-7 and save the results in the dataframe
"""
import logging as log
import argparse
import os

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import time
import cvxpy as cp

from eval_utils import Evaluator
from utils import make_q_learning_operator, make_adv_q_learning, bounded_successive_approximation, default_termination
from utils import reward_direct_policy_evaluation, cost_direct_policy_evaluation
from hcpi_utils import compute_approx_hopt

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
# LAMBDA_C_vals = [1.0, 0.0]
LAMBDA_C_vals = [1.0, 0.0, 0.1, 0.01,]
LAMBDA_COEFFS = [(lr, lc) for lc in LAMBDA_C_vals for lr in LAMBDA_R_vals]

# ----------- HPCI parameters -----------
DELTA_HCPI = [0.1, 0.3, 0.5, 0.7, 0.9]
# run q-learning w/ advantage
Q_LEARNING_ADV_MODES = [True]
HOPT_OPE_METHOD = ['DR', 'WDR']
LOWER_BOUND_METHOD = ['t_test'] # confidence interval gives very high bounds so removing them for now
Q_LEARNING_ITERATIVE_MODE = False

# -------- User inputs -------------
parser = argparse.ArgumentParser(description="For running the H-OPT algotihms")

parser.add_argument(
    '--seed', type=int, default=0, help='the seed that defines the input data to work with ')
parser.add_argument(
    '--freq', type=float, default=10.0, help='the frequency that determines what counts as a rare action')
parser.add_argument(
    '--cost', type=float, default=10.0, help='the cost for taking rare action')
parser.add_argument(
    '--n_bootstrap', type=int, default=10, help='the cost for taking rare action')

args = parser.parse_args()

SEED = args.seed
FREQUENCY_THRESHOLD = args.freq
COST_FOR_RARE_DECISION = args.cost

# -------- Folder Paths -------------
basepath = '/enter/path/here'

# Path variables
LOG_PATH = f'{basepath}/logs/hopt'
IMPORT_PATH = f'{basepath}/m_hat/{SEED}'
IMPORT_PI_PATH = f'{basepath}/output/{SEED}/freq_{FREQUENCY_THRESHOLD}_cost_{COST_FOR_RARE_DECISION}'
EXPORT_PATH = f'{IMPORT_PI_PATH}/hopt'

if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

# -------- Log -------------
log.basicConfig(
    filename=f'{LOG_PATH}/{SEED}-freq_{FREQUENCY_THRESHOLD}_cost_{COST_FOR_RARE_DECISION}-hcpi.log', filemode='w',
    format='%(asctime)s - %(levelname)s \t %(message)s',
    level=log.DEBUG)

# ---------    Main code starts here --------------
log.info("Starting the HOPT script")

results = []


# ------------------------------------------------------------
#                  5_7-HCPI
# ------------------------------------------------------------
log.info("----- 5-7 HCPI ------")

log.info("Loading m_hat matrices")
P_mat, R_mat = pickle.load(open(f"{IMPORT_PATH}/MDP_mat.p", "rb"))

# load the defaults for this seed
C_mat = pickle.load(open(f"{IMPORT_PI_PATH}/C_mat.p", "rb"))
pi_baseline = pickle.load(open(f"{IMPORT_PI_PATH}/pi_baseline.p", "rb"))

# check if the same cost was used for both runs
assert COST_FOR_RARE_DECISION == np.max(C_mat)

# create a random policy
random_policy = np.ones((nS+2, nA))/nA

# get R_sa once
R_sa = np.einsum('sat,sat -> sa', R_mat, P_mat)

# create the evaluator object
evaluator = Evaluator(gamma=gamma,
                      pi_baseline=pi_baseline,
                      C_mat=C_mat,
                      cost_for_rare_decision=COST_FOR_RARE_DECISION,
                      n_bootstrap=args.n_bootstrap, )


# ------------------------------------------------------------
#         Load the datasets
# ------------------------------------------------------------

# --- Train set ---
traj_tr = pickle.load(open(f'{IMPORT_PATH}/trajDr_tr.pkl', 'rb'))
log.info(f'Size of train set before processing {len(traj_tr)}')
train_trajectories = evaluator.preprocess_trajecteories(traj_tr)
log.info(f'Size of train set after processing {len(train_trajectories)}')
# get the mean
train_mean_stats = evaluator.get_mean_stats(train_trajectories)
log.info(f'--- Mean stats --- \n R: {train_mean_stats[0]}, C: {train_mean_stats[1]}')

# --- Test set ---
log.info("Loading test set")
traj_te = pickle.load(open(f'{IMPORT_PATH}/trajDr_te.pkl', 'rb'))
test_trajectories = evaluator.preprocess_trajecteories(traj_te)
N_test = len(test_trajectories)
log.info(f'Effective sample size of test set {N_test}')
# get the mean
test_mean_stats = evaluator.get_mean_stats(test_trajectories)
log.info(f'--- Mean stats --- \n R: {test_mean_stats[0]}, C: {test_mean_stats[1]}')

# --- Validation set ---
log.info("Loading validation set")
traj_va = pickle.load(open(f'{IMPORT_PATH}/trajDr_va.pkl', 'rb'))
va_trajectories = evaluator.preprocess_trajecteories(traj_va)
log.info(f'Effective sample size of validation set {len(va_trajectories)}')
# get the mean
va_mean_stats = evaluator.get_mean_stats(va_trajectories)
log.info(f'--- Mean stats --- \n R: {va_mean_stats[0]}, C: {va_mean_stats[1]}')

# the max C value required for clipping and normalizing purpopses later in HOPT
va_C_max = (max(len(traj) for traj in va_trajectories)) * COST_FOR_RARE_DECISION


# ------------------------------------------------------------
#          Helper function for OPE estimation
# ------------------------------------------------------------

def ope_estimation_on_validation(pi_solution, method='DR'):
    """
    Returns the OPE estimates as a list on the validation set

    :param pi_solution: the target policy for OPE
    :param method: what kind of OPE method to use - DR/WDR

    :return: list of ope estimates for validation set
    """
    # estimate Q_hat
    # R
    rV_sol = reward_direct_policy_evaluation(P_mat, R_mat, gamma, pi_solution)
    rQ_sol = R_sa + gamma * np.einsum('sat,t -> sa', P_mat, rV_sol)
    # C
    cV_sol = cost_direct_policy_evaluation(P_mat, C_mat, gamma, pi_solution)
    cQ_sol = C_mat + gamma * np.einsum('sat,t -> sa', P_mat, cV_sol)

    if method == 'DR':
        R_list, C_list = evaluator.list_format_doubly_robust_ope(va_trajectories, pi_e=pi_solution, rQ_e=rQ_sol,
                                                                 cQ_e=cQ_sol)
    elif method == 'WDR':
        R_list, C_list = evaluator.list_format_weighted_doubly_robust_ope(va_trajectories, pi_e=pi_solution,
                                                                          rQ_e=rQ_sol, cQ_e=cQ_sol)

    return R_list, C_list


def ope_estimation(pi_solution, alg_name, lR, lC, delta_hcpi, lower_bound_strategy):
    """
    helper function to do DR estimation and store the results in the list

    Note: does only for test set as validation set is already accounted for safety-test in HCPI

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
    dr_mean_stats, _, _ = evaluator.doubly_robust_ope(test_trajectories,
                                                      pi_e=pi_solution,
                                                      rQ_e=rQ_sol,
                                                      cQ_e=cQ_sol,
                                                      )

    log.info('-- (TEST) Mean stats ---')
    log.info('-- Doubly Robust (DR) ---')
    log.info(f'R: {dr_mean_stats[0]}, C: {dr_mean_stats[1]}')

    # save to the results list
    results.append([SEED, alg_name,  # seed, alg_name
                    lR, lC,  # 'lambda_R', 'lambda_C',
                    'test',  # 'dataset',#train/tes/valid
                    dr_mean_stats[0], dr_mean_stats[1],  # 'R_mean', 'C_mean',
                    None, None,   # 'R_boot_mean', 'R_boot_std',
                    None, None,  # 'C_boot_mean', 'C_boot_std',
                    pi_diff,
                    None, # eps
                    None, # delta
                    delta_hcpi,  # 'delta_hcpi',
                    'DR',  # 'ope-estimator',
                    lower_bound_strategy,  # lower_bound_strategy
                    ])

    # ------- Weighted Doubly Robust OPE ---------------
    dr_mean_stats, _, _ = evaluator.weighted_doubly_robust_ope(test_trajectories,
                                                               pi_e=pi_solution,
                                                               rQ_e=rQ_sol,
                                                               cQ_e=cQ_sol,
                                                               )

    log.info('-- Weighted Doubly Robust (WDR) ---')
    log.info(f'R: {dr_mean_stats[0]}, C: {dr_mean_stats[1]}')

    # save to the results list
    results.append([SEED, alg_name,  # seed, alg_name
                    lR, lC,  # 'lambda_R', 'lambda_C',
                    'test',  # 'dataset',#train/tes/valid
                    dr_mean_stats[0], dr_mean_stats[1],  # 'R_mean', 'C_mean',
                    None, None,   # 'R_boot_mean', 'R_boot_std',
                    None, None,  # 'C_boot_mean', 'C_boot_std',
                    pi_diff,
                    None, # eps
                    None, # delta
                    delta_hcpi,  # 'delta_hcpi',
                    'WDR',  # 'ope-estimator',
                    lower_bound_strategy,  # lower_bound_strategy
                    ])



log.info(f'Starting the HOPT solutions')
# ------------------------------------------------------------
#               Start the actual H-PT
# ------------------------------------------------------------

for coeffs in LAMBDA_COEFFS:
    log.info(f'Staring with \lambda_R={coeffs[0]} \lambda_C={coeffs[1]}')
    # for each lambda_R, lambda_C combination test different kind of surrogates Qs
    for q_learn_w_advantage in Q_LEARNING_ADV_MODES:
        log.info(f'Q-learning-w-adv mode is set to {q_learn_w_advantage}')
        # ------- Find pi-hat first ---
        pi_hat = None

        if Q_LEARNING_ITERATIVE_MODE:
            raise NotImplementedError("lower in the priority!")
        else:
            if q_learn_w_advantage:
                # use Q-learning w/ advantage constraints
                q_learning_operator = make_adv_q_learning(P=P_mat, R=R_mat, C=C_mat, discount=gamma,
                                                          baseline=pi_baseline,
                                                          coeffs=coeffs, )
            else:
                # do greedy Q-lambda learning
                q_learning_operator = make_q_learning_operator(P=P_mat, R=R_mat, C=C_mat, discount=gamma,
                                                               coeffs=coeffs, )

            # call Q-learning operator here
            start_time = time.time()
            # find the optimal pi
            pi_hat = bounded_successive_approximation(random_policy,
                                                      operator=q_learning_operator,
                                                      termination_condition=default_termination,
                                                      max_limit=10, )

            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            log.info(f"pi-hat found in in {time_elapsed}")

        assert pi_hat is not None

        # ------- Do the approximation for HOPT here ---
        # Test different hopt hyper-params for this lambda and pi_hat result
        for ope_method in HOPT_OPE_METHOD:
            for lower_bound_strategy in LOWER_BOUND_METHOD:
                log.info(f'Starting HOPT with OPE method: {ope_method} and lower bound method: {lower_bound_strategy}')
                for delta in DELTA_HCPI:

                    # find the pi_hcpi here
                    pi_hcpi, pi_reg = compute_approx_hopt(pi_b=pi_baseline,
                                                          pi_hat=pi_hat,
                                                          ope_estimation_on_validation=ope_estimation_on_validation,
                                                          confidence=delta/2., #due to union bound
                                                          train_mean_stats=train_mean_stats,
                                                          ope_method=ope_method,
                                                          lower_bound_strategy=lower_bound_strategy,
                                                          C_min=0, C_max=va_C_max, )


                    # evaluate and store the results
                    alg_name = f'hopt-{ope_method}'
                    if q_learn_w_advantage:
                        alg_name += '-adv'

                    log.info(f'Evaluating the solution for {alg_name} with delta={delta} and regularization:{pi_reg}')

                    # does the ope estimation heavy-lifting
                    ope_estimation(pi_solution=pi_hcpi, alg_name=alg_name, lR=coeffs[0], lC=coeffs[1],
                                   delta_hcpi=delta, lower_bound_strategy=lower_bound_strategy)

# -----------------------------------------------------------------------
#        Save everything as pandas df
# -----------------------------------------------------------------------
log.info(f'Saving the results')

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

df.to_csv(path_or_buf=f'{EXPORT_PATH}/results.csv')
df.to_pickle(path=f'{EXPORT_PATH}/results.pkl')

# ------- Done ----------------
log.info("Done")
