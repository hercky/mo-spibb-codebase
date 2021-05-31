"""
For running the experiments with tuned hyper-parameters

python -W ignore scripts/benchmark_param_agents.py --out_dir {OUT_DIR}  --param_path {PARAM_PATH} --num_runs {NUM_RUNS} --exp_name {EXP_NAME}
"""
#!/usr/bin/env python

import argparse

from gridworld.agents.tabular_spibb_agents import ConstSPIBBAgent
from gridworld.agents.tabular_hcpi_agents import HCPIAgent
from gridworld.agents.tabular_base_agent import RewardShapingPIAgent, ReshapedAdvantagePIAgent, UnconstPIAgent

from gridworld.core.logx import setup_logger_kwargs
from gridworld.core.utils import default_termination
from gridworld.core.benchmark_param_agents import benchmark_param_based_agents

import numpy as np


# hyper-parameters to test for goes over here
AGENT_LIST = ["s-opt", "h-opt", "reg-PI", "rs-PI", "adv-rs-PI"]


# Dataset generation hyper-parameters
NB_TRAJ_LIST = [10, 50, 500, 2000]
RATIO_LIST  = [0.1, 0.4, 0.7, 0.9]

# Prepare coefficients list
# Note: all coeffs need to be >=0
LAMBDA_R_vals = [1.0, 0.0]
LAMBDA_C_vals = [1.0, 0.0]
LAMBDA_COEFFS = [(lr, lc) for lc in LAMBDA_C_vals for lr in LAMBDA_R_vals]

# the delta for spibb eq (and similar to DELTA_HCPI)
DELTA=0.1

if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    # env specific arguments
    parser.add_argument('--env_name', type=str, default='large_grid-200')
    parser.add_argument('--env_tries', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_lim', type=float, default=2.0)
    # exp logging specific args
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='benchmark_params')
    parser.add_argument('--out_dir', type=str, default='/tmp/best-cmdp/')
    parser.add_argument('--param_path', type=str, default='./best_params.p')
    parser.add_argument('--num_runs', type=int, default=10)
    # parse args
    args = parser.parse_args()

    # append the env_name to the exp_name

    # Create the agents
    exp_agents = []
    for agent_type in AGENT_LIST:

        # Prepare agent
        if agent_type == 's-opt':
            # add the SOP agent to the mix
            agent_kwargs = dict(termination_condition=default_termination,
                                coeff_list=LAMBDA_COEFFS)
            agent = ConstSPIBBAgent(**agent_kwargs)
            exp_agents.append(agent)
        elif agent_type == 'h-opt':
            # add different HOPT agents
            agent_kwargs = dict(lower_bound_strategy="student_t_test",
                                estimator_type="doubly_robust", #tmp names
                                coeff_list=LAMBDA_COEFFS,
                                training_size=0.7,
                                )
            agent = HCPIAgent(**agent_kwargs)
            exp_agents.append(agent)
        elif agent_type == 'reg-PI':
            agent_kwargs = dict(termination_condition=default_termination, )
            agent = UnconstPIAgent(**agent_kwargs)
            exp_agents.append(agent)
        elif agent_type == 'rs-PI':
            agent_kwargs = dict(termination_condition=default_termination,
                                coeff_list=LAMBDA_COEFFS)
            agent = RewardShapingPIAgent(**agent_kwargs)
            exp_agents.append(agent)
        elif agent_type == 'adv-rs-PI':
            agent_kwargs = dict(termination_condition=default_termination,
                                coeff_list=LAMBDA_COEFFS)
            agent = ReshapedAdvantagePIAgent(**agent_kwargs)
            exp_agents.append(agent)
        else:
            raise Exception("not implemented yet")

    # Starting the experiment now
    print(f"Starting the experiment: {args.exp_name}")

    # Run all baselines for only one env
    env_name = args.env_name
    print(f"Environment: {env_name}")

    # Prepare a logger
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name,
                                        env_name=env_name,
                                        seed=args.seed,
                                        data_dir=args.out_dir,
                                        print_along=False,
                                        timestamp=True)

    # benchmark the performance of each agent on an instance of this env
    benchmark_param_based_agents(args.env_name,
                                 num_runs=args.num_runs,
                                 agent_param_dict_path=args.param_path,
                                 num_creation_tries=args.env_tries,
                                 agent_list=exp_agents,
                                 seed=args.seed,
                                 #
                                 delta=DELTA,
                                 nb_trajectories_list=NB_TRAJ_LIST,
                                 ratio_list=RATIO_LIST,
                                 discount=args.gamma,
                                 cost_limit=args.cost_lim,
                                 logger_kwargs=logger_kwargs,
                                 )

    print(f"Experiment finished")