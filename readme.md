## Multi-Objective SPIBB: Seldonian Offline PolicyImprovement with Safety Constraints in Finite MDPs

Accompanying codebase for MO-SPIBB

### Setup

Create a conda environment using the `requirements.txt` to install all the dependent libraries.

```
$ conda create --name <env> --file <path/to/requirements.txt>
```


### Gridworld Experiments 


The code for the synthetic experiments (Section 4) can be found in `gridworld/`
Notes for running the scripts:

* First add the current path to the python path, 
    ```
    export PYTHONPATH=$PYTHONPATH:/path/to/gridworld
    ``` 
* Run the corresponding scripts in `gridworld/scripts/` to launch an experiment. For instance,
    ```
    python -W ignore scripts/delta_agents.py --out_dir {OUT_DIR}  --exp_name {EXP_NAME} --num_runs {NUM_RUNS}
    ```
* The plotting notebooks for generating the figures are provided in `gridworld/plots/`.  


### Sepsis Experiments


The code for sepsis experiments (Section 5) can be found in `sepsis/` . Note: fix the `basepath` variable in all these scripts to the corresponding folder locations. 
Follow, the following instructions for running the scripts:

#### Step 1 - Cohort and Pre-processing

First, follow the Step 1 (pre-processing and cohort design) as described by: https://github.com/MLD3/RL-Set-Valued-Policy/tree/master/mimic-sepsis . This part uses the code from (Tang et al., 2020; Komorowski et al., 2018) for MIMIC dataset.


#### Step 2 - Dataset creation
* Run the notebook `sepsis/1_preprocess/1_1_process_interventions.ipynb` to discretize the action space. 
* Launch the pre-processing script `1_preprocess/run.py` to discretize the state space and create the train/valid/test splits. 


#### Step 3 - MDP estimation

* Run the scripts in `2_opt/mdp_estimation.py` to estimate the MLE MDP and the baseline policy. For instance, 
    ```
    python mdp_estimation.py --seed {SEED}
    ```

#### Step 4 - Run different agents


* For running Linearized, Adv-Linearized and S-OPT agents, run the scripts in `2_opt/run_pi.py`. This will save the solution policies in the folder defined by `basepath` variable in the script. 
* Next, evaluate the performance of these agents via `2_opt/run_ope.py` scripts. 
* For running H-OPT agents follow the instruction in ` 2_opt/run_hopt.py` (does both the optimization and OPE in the same script).


Note: The qualitative analysis claims can be found in the notebooks `2_opt/6-0_qual_analysis_aggressive_treatment.ipynb` and `vim 2_opt/6-1_rare_action_freq_100.ipynb`.

