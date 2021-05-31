"""
Does the env-model estimation and clinician policy estimation.

Assumes, the steps in 1_preprocess have been completed before

Usage:
- for single run
python mdp_estimation.py --seed {SEED}

- for 10 runs
for VAR in $(seq 0 9); do python mdp_estimation.py --seed $VAR ; done
"""


import logging as log
import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm

# ----------- Defaults -------------
nS, nA = 750, 25
DEATH_STATE = 750
SURVIVAL_STATE = 751


# -------- Setting up -------------
parser = argparse.ArgumentParser(description="For running the PI methods with constraints")

parser.add_argument(
    '--seed', type=int, default=0, help='the seed that defines the intput data to work with ')

# script
args = parser.parse_args()

SEED = args.seed

# main folder path
basepath = 'enter/path/here'
# Path variables
LOG_PATH = f'{basepath}/logs'
IMPORT_PATH = f'{basepath}/processed/{SEED}'
EXPORT_PATH = f'{basepath}/m_hat/{SEED}'

if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

# set up the log config
log.basicConfig(
        filename=f'{LOG_PATH}/{SEED}-mdp-estimation.log', filemode='w',
        format='%(asctime)s - %(levelname)s \t %(message)s',
        level=log.DEBUG)

# ---------    Main code starts here --------------
log.info("Starting the mdp-estimation now")

# ------------------------------------------------------------
#                  5_0-Fix reward
# ------------------------------------------------------------
log.info("----- 5-0_fix_reward ------")

# For the train set
traj_tr = pickle.load(open(f'{IMPORT_PATH}/trajD_tr.pkl', 'rb'))

for traj in traj_tr:
    if traj[-1]['r'] == 0:
        traj[-1]['r'] = -100
    if traj[-1]['r'] == 1:
        traj[-1]['r'] = 100

pickle.dump(traj_tr, open(f'{EXPORT_PATH}/trajDr_tr.pkl', 'wb'))

# For validation set
traj_va = pickle.load(open(f'{IMPORT_PATH}/trajD_va.pkl', 'rb'))

for traj in traj_va:
    if traj[-1]['r'] == 0:
        traj[-1]['r'] = -100
    if traj[-1]['r'] == 1:
        traj[-1]['r'] = 100

pickle.dump(traj_va, open(f'{EXPORT_PATH}/trajDr_va.pkl', 'wb'))

# For test set
traj_te = pickle.load(open(f'{IMPORT_PATH}/trajD_te.pkl', 'rb'))

for traj in traj_te:
    if traj[-1]['r'] == 0:
        traj[-1]['r'] = -100
    if traj[-1]['r'] == 1:
        traj[-1]['r'] = 100

pickle.dump(traj_te, open(f'{EXPORT_PATH}/trajDr_te.pkl', 'wb'))


train_death_ratio = np.mean([ traj[-1]['r'] == -100 for traj in traj_tr])
validation_death_ratio  = np.mean([traj[-1]['r'] == -100 for traj in traj_va])
test_death_ratio = np.mean([traj[-1]['r'] == -100 for traj in traj_te])

log.info(f'The percentage of deaths in train set {train_death_ratio}')
log.info(f'The percentage of deaths in validation set {validation_death_ratio}')
log.info(f'The percentage of deaths in test set {test_death_ratio}')

# ------------------------------------------------------------
#                  5_1-Env model
# ------------------------------------------------------------
log.info("----- 5-1_env_model ------")

#traj_tr = pickle.load(open(f'{EXPORT_PATH}/trajDr_tr.pkl', 'rb'))

log.info("Estimating P matrix")
# count but save in matrices now
trans_counts_mat = np.zeros((nS + 2, nA, nS + 2))

# use only training trajectories
for trajectory in tqdm(traj_tr):
    for t, transition in enumerate(trajectory):
        s = transition['s']
        a = transition['a']
        r = transition['r']
        s_ = transition['s_']
        if s_ is None:
            if r == -100:
                s_ = DEATH_STATE  # death
            elif r == 100:
                s_ = SURVIVAL_STATE  # survival
            else:
                raise NotImplementedError

        trans_counts_mat[s, a, s_] += 1

# Modify the counts
orig_counts_mat = trans_counts_mat.copy()

# assign absorbing states
assert trans_counts_mat[DEATH_STATE, :, :].sum() == 0
assert trans_counts_mat[SURVIVAL_STATE, :, :].sum() == 0
# Add the death / life absorbing state
trans_counts_mat[DEATH_STATE, :, DEATH_STATE] = 1
trans_counts_mat[SURVIVAL_STATE, :, SURVIVAL_STATE] = 1

# Note: *Not in original paper* send any unobserved actions to death
no_tx_idx = trans_counts_mat.sum(axis=-1) == 0
trans_counts_mat[no_tx_idx, DEATH_STATE] = 1

# Normalise the transition counts
# Build probabilistic MDP model

# Convert counts into probability
P_mat = trans_counts_mat / trans_counts_mat.sum(axis=-1, keepdims=True)
assert np.allclose(1, P_mat.sum(axis=-1))


log.info("Estimating R matrix")
# Fill the R_mat here
R_mat = np.zeros((nS + 2, nA, nS + 2))
R_mat[..., DEATH_STATE] = -100.0  # death
R_mat[..., SURVIVAL_STATE] = 100.0  # survival

# No reward once in absorbing state, because can't take any actions there
R_mat[DEATH_STATE, ..., DEATH_STATE] = 0
R_mat[SURVIVAL_STATE, ..., SURVIVAL_STATE] = 0

log.info("Saving the MDP model now.")
with open(f'{EXPORT_PATH}/MDP_mat.p', 'wb') as f:
    pickle.dump((P_mat, R_mat), f)

with open(f'{EXPORT_PATH}/MDP_counts.p', 'wb') as f:
    pickle.dump(orig_counts_mat, f)


# ------------------------------------------------------------
#                  5_2-Clinician policy
# ------------------------------------------------------------
log.info("----- 5-2_clinician policy ------")

# traj_tr = pickle.load(open(f'{EXPORT_PATH}/trajDr_tr.pkl', 'rb'))
# Count frequency of each transition
policy_counts = np.zeros((nS, nA), dtype=int)
for trajectory in tqdm(traj_tr):
    for t, transition in enumerate(trajectory):
        s = transition['s']
        a = transition['a']
        policy_counts[s][a] += 1

# Note: trans_count_mat is what used to be orig_counts_mat
trans_count_mat = pickle.load(open(f'{EXPORT_PATH}/MDP_counts.p', 'rb'))

uf_pi_mat = np.zeros((nS, nA))

# adjusted counts
for trajectory in tqdm(traj_tr):
    for t, transition in enumerate(trajectory):
        s = transition['s']
        a = transition['a']
        # every action is accounted now
        uf_pi_mat[s][a] += 1

# normalization
filter_states = uf_pi_mat.sum(axis=-1) > 0
uf_pi_mat[filter_states] /= uf_pi_mat[filter_states].sum(axis=-1, keepdims=True)

# save it here
with open(f'{EXPORT_PATH}/clinician_policy_mat.p', 'wb') as f:
    pickle.dump(uf_pi_mat, f)


# ------- Done ----------------
log.info("Done")


