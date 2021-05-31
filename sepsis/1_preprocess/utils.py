
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm
import seaborn as sns
from joblib import dump, load
import pandas as pd
import argparse


binary_fields = ['gender','mechvent','re_admission']

norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
    'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
    'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
    'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',
    'PaO2_FiO2','cumulated_balance', 'elixhauser', 'Albumin', u'CO2_mEqL', 'Ionised_Ca']

log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',
              'input_total','input_4hourly','output_total','output_4hourly', 'bloc']



# define an action mapping - how to get an id representing the action from the (iv,vaso) tuple
action_map = {(iv, vaso): 5*iv+vaso for iv in range(5) for vaso in range(5)}

def make_trajectories(df):
    trajectories = []
    for i, g in tqdm(df.groupby('icustayid')):
        try:
            g = g.reset_index(drop=True)
            trajectory = []
            for t in range(len(g) - 1):
                transition = {
                    's': g.loc[t, 'state'],
                    'a': action_map[
                        int(g.loc[t, 'iv_input_NEW']),
                        int(g.loc[t, 'vaso_input_NEW'])
                    ],
                    'r': g.loc[t, 'reward'],
                    's_': g.loc[t + 1, 'state'],
                    'a_': action_map[
                        int(g.loc[t + 1, 'iv_input_NEW']),
                        int(g.loc[t + 1, 'vaso_input_NEW'])
                    ],
                    'done': False,
                }
                trajectory.append(transition)

            t = len(g) - 1
            trajectory.append({
                's': g.loc[t, 'state'],
                'a': action_map[
                    int(g.loc[t, 'iv_input_NEW']),
                    int(g.loc[t, 'vaso_input_NEW'])
                ],
                'r': g.loc[t, 'reward'],
                's_': None,
                'a_': None,
                'done': True,
            })
            #             print(trajectory)
            trajectories.append(trajectory)
        except:
            print(i)
            # display(g)
            raise
    return trajectories