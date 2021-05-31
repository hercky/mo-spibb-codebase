"""
This code does the entire pre-processing by combining the
individual notebooks in https://github.com/MLD3/RL-Set-Valued-Policy/tree/master/mimic-sepsis/mimic_sepsis_rl/1_preprocess

Run the notebook 1_1_process_interventions first before running this code
"""

import logging as log
import argparse
import numpy as np
import pandas as pd
import random
import pickle
import pickle as pkl
import os
import copy
from pandas import DataFrame
from sklearn.cluster import KMeans
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
from tqdm import tqdm


from utils import binary_fields, norm_fields, log_fields, make_trajectories

# -------- Setting up -------------
parser = argparse.ArgumentParser(description="Main pre-processing script")

parser.add_argument(
    '--seed', type=int, default=0, help='the random seed to initialize with')


# script
args = parser.parse_args()

SEED = args.seed

# main folder path
basepath = '/enter/path/here'

# Path variables
DATA_PATH = f'{basepath}/data'
LOG_PATH = f'{basepath}/logs'

EXPORT_PATH = f'{basepath}/processed/{SEED}'
if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

    
# set up the log config
log.basicConfig(
        filename=f'{LOG_PATH}/{SEED}-preprocess.log', filemode='w',
        format='%(asctime)s - %(levelname)s \t %(message)s',
        level=log.DEBUG)

# ---------    Main code starts here --------------
log.info("Starting the pre-processing now")

# ------------------------------------------------------------
#                  1_2_preprocess
# ------------------------------------------------------------
log.info("----- 1_2_preprocess ------")
# This notebook reads in the discretised input data and then preprocesses the model features
# Firstly, values deemed excessively high/low are capped
# Relevant binary features and normally/log-normally features are standardised accordingly
# Training and test sets are split - 70% train, 10% validation, 20% test
# Resulting datasets are saved to file.

# load the data
disc_inp_data = pd.read_csv(f"{DATA_PATH}/discretised_input_data.csv")

# add rewards - sparsely for now; reward function shaping comes in a separate script
# if died in hospital -> reward at final timestep = 0
# if survived in hospital -> reward at final timestep = 1
disc_inp_data['terminal_reward'] = 0
m = ~disc_inp_data.duplicated('icustayid', 'last') # select the last timestep for each patient
disc_inp_data.loc[m, 'terminal_reward'] = 1-disc_inp_data.loc[m, 'died_in_hosp']

# now split into train/validation/test sets
random.seed(42 + SEED)
unique_ids = disc_inp_data['icustayid'].unique()
random.shuffle(unique_ids)
train_sample = 0.7
val_sample = 0.1
test_sample = 0.2
train_num = int(len(unique_ids) * 0.7)
val_num = int(len(unique_ids)*0.1) + train_num
train_ids = unique_ids[:train_num]
val_ids = unique_ids[train_num:val_num]
test_ids = unique_ids[val_num:]


train_set = disc_inp_data.loc[disc_inp_data['icustayid'].isin(train_ids)].copy()
val_set = disc_inp_data.loc[disc_inp_data['icustayid'].isin(val_ids)].copy()
test_set = disc_inp_data.loc[disc_inp_data['icustayid'].isin(test_ids)].copy()


# normalise binary fields
train_set[binary_fields] = train_set[binary_fields] - 0.5
val_set[binary_fields] = val_set[binary_fields] - 0.5
test_set[binary_fields] = test_set[binary_fields] - 0.5

# normal distn fields
mean_stds = {}
for item in norm_fields:
    av = train_set[item].mean()
    std = train_set[item].std()
    train_set[item] = (train_set[item] - av) / std
    val_set[item] = (val_set[item] - av) / std
    test_set[item] = (test_set[item] - av) / std
    mean_stds[item] = (av,std)


# log normal fields
mean_stds_log = {}
train_set[log_fields] = np.log(0.1 + train_set[log_fields])
val_set[log_fields] = np.log(0.1 + val_set[log_fields])
test_set[log_fields] = np.log(0.1 + test_set[log_fields])
for item in log_fields:
    av = train_set[item].mean()
    std = train_set[item].std()
    train_set[item] = (train_set[item] - av) / std
    val_set[item] = (val_set[item] - av) / std
    test_set[item] = (test_set[item] - av) / std
    mean_stds_log[item] = (av,std)


# print head
log.info(str(train_set.head()))

# save
pickle.dump({'norm': mean_stds, 'lognorm': mean_stds_log}, open(f'{EXPORT_PATH}/normalization_params.p', 'wb'))

train_set.to_csv(f'{EXPORT_PATH}/rl_train_set_unscaled.csv',index = False)
val_set.to_csv(f'{EXPORT_PATH}/rl_val_set_unscaled.csv', index = False)
test_set.to_csv(f'{EXPORT_PATH}/rl_test_set_unscaled.csv', index = False)

# scale features to [0,1] in train set, similar in val and test

scalable_fields = copy.deepcopy(binary_fields)
scalable_fields.extend(norm_fields)
scalable_fields.extend(log_fields)
for col in scalable_fields:
    minimum = min(train_set[col])
    maximum = max(train_set[col])
    train_set[col] = (train_set[col] - minimum)/(maximum-minimum)
    val_set[col] = (val_set[col] - minimum)/(maximum-minimum)
    test_set[col] = (test_set[col] - minimum)/(maximum-minimum)

# print after scaling
log.info(str(train_set.head()))


# save scaled also
train_set.to_csv(f'{EXPORT_PATH}/rl_train_set_scaled.csv',index = False)
val_set.to_csv(f'{EXPORT_PATH}/rl_val_set_scaled.csv', index = False)
test_set.to_csv(f'{EXPORT_PATH}/rl_test_set_scaled.csv', index = False)

# ------------------------------------------------------------
#                  1_3_Clustering
# ------------------------------------------------------------
log.info("----- 1_3_clustering_------")
# load the correct dataset
train_data = pd.read_csv(f"{EXPORT_PATH}/rl_train_set_unscaled.csv")
val_data = pd.read_csv(f"{EXPORT_PATH}/rl_val_set_unscaled.csv")
test_data = pd.read_csv(f"{EXPORT_PATH}/rl_test_set_unscaled.csv")


with open("state_features.txt") as f:
    feat = f.read()
cluster_features = feat.split()


# extract out the features from data
train_cluster_data = train_data[cluster_features]
val_cluster_data = val_data[cluster_features]
test_cluster_data = test_data[cluster_features]

# Do K-means here
kmeans_train = KMeans(n_clusters=750, random_state=0, verbose=10, n_init=10, n_jobs=-1).fit(train_cluster_data)


# Save the results
dump(kmeans_train, f'{EXPORT_PATH}/kmeans_750.joblib')

log.info(f"K-mean intertia is {kmeans_train.inertia_}")

# clusters for the train data
train_clusters = kmeans_train.labels_

# here, compute cluster centroids for val and test data
val_clusters = kmeans_train.predict(val_cluster_data)
test_clusters = kmeans_train.predict(test_cluster_data)

train_data['state'] = train_clusters
val_data['state'] = val_clusters
test_data['state'] = test_clusters


# Create dataframes

# Train
train_set_final = DataFrame()
train_set_final['bloc'] = train_data['bloc']
train_set_final['icustayid'] = train_data['icustayid']
train_set_final['state'] = train_data['state']
train_set_final['reward'] = train_data['terminal_reward']
train_set_final['mortality'] = train_data['died_in_hosp']

train_set_final['vaso_input'] = train_data['vaso_input']
train_set_final['iv_input'] = train_data['iv_input']
train_set_final['vaso_input_NEW'] = train_data['vaso_input_NEW']
train_set_final['iv_input_NEW'] = train_data['iv_input_NEW']

# Valid
val_set_final = DataFrame()
val_set_final['bloc'] = val_data['bloc']
val_set_final['icustayid'] = val_data['icustayid']
val_set_final['state'] = val_data['state']
val_set_final['reward'] = val_data['terminal_reward']
val_set_final['mortality'] = val_data['died_in_hosp']

val_set_final['vaso_input'] = val_data['vaso_input']
val_set_final['iv_input'] = val_data['iv_input']
val_set_final['vaso_input_NEW'] = val_data['vaso_input_NEW']
val_set_final['iv_input_NEW'] = val_data['iv_input_NEW']

# Test
test_set_final = DataFrame()
test_set_final['bloc'] = test_data['bloc']
test_set_final['icustayid'] = test_data['icustayid']
test_set_final['state'] = test_data['state']
test_set_final['reward'] = test_data['terminal_reward']
test_set_final['mortality'] = test_data['died_in_hosp']

test_set_final['vaso_input'] = test_data['vaso_input']
test_set_final['iv_input'] = test_data['iv_input']
test_set_final['vaso_input_NEW'] = test_data['vaso_input_NEW']
test_set_final['iv_input_NEW'] = test_data['iv_input_NEW']


# Save
train_set_final.to_csv(f'{EXPORT_PATH}/rl_train_data_discrete.csv',index=False)
val_set_final.to_csv(f'{EXPORT_PATH}/rl_val_data_discrete.csv', index=False)
test_set_final.to_csv(f'{EXPORT_PATH}/rl_test_data_discrete.csv',index=False)

# ------------------------------------------------------------
#                  1_END_SANITY_CHECK
# ------------------------------------------------------------
log.info(f"Sanity check starting!")

df_train = pd.read_csv(f'{EXPORT_PATH}/rl_train_set_scaled.csv')
df_val =  pd.read_csv(f'{EXPORT_PATH}/rl_val_set_scaled.csv')
df_test = pd.read_csv(f'{EXPORT_PATH}/rl_test_set_scaled.csv')

feature_cols = binary_fields + norm_fields + log_fields
log.info(f"len of feature columns {len(feature_cols)}")

X_tr = df_train[feature_cols].values
X_va = df_val[feature_cols].values
X_te = df_test[feature_cols].values
y_tr = df_train['died_in_hosp'].values
y_va = df_val['died_in_hosp'].values
y_te = df_test['died_in_hosp'].values

clf = LogisticRegression(random_state=0)
clf.fit(X_tr, y_tr)

log.info(f"Sanity check {metrics.roc_auc_score(y_va, clf.predict_proba(X_va)[:,1])}")


# ------------------------------------------------------------
#                  1_Z_REFORMATTING
# ------------------------------------------------------------
log.info(f"Creating trajectories")

# VAL
traj_va = make_trajectories(pd.read_csv(f'{EXPORT_PATH}/rl_val_data_discrete.csv'))
pkl.dump(traj_va, open(f'{EXPORT_PATH}/trajD_va.pkl', 'wb'))

# TEST
traj_te = make_trajectories(pd.read_csv(f'{EXPORT_PATH}/rl_test_data_discrete.csv'))
pkl.dump(traj_te, open(f'{EXPORT_PATH}/trajD_te.pkl', 'wb'))


# TRAIN
traj_tr = make_trajectories(pd.read_csv(f'{EXPORT_PATH}/rl_train_data_discrete.csv'))
pkl.dump(traj_tr, open(f'{EXPORT_PATH}/trajD_tr.pkl', 'wb'))


# ------- Done ----------------
log.info("Done")


