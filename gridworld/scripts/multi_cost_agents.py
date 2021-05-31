"""
for running the scaling experiments in Appendix D

usage:
"python -W ignore scripts/multi_cost_agents.py --out_dir {OUT_DIR} --exp_name {EXP_NAME}  --num_runs {NUM_RUNS}
"""
#!/usr/bin/env python

import argparse
import numpy as np

from gridworld.agents.multi_cost_agents import MultiCostHCPIAgent, MultiConstSPIBBAgent
from gridworld.core.multi_cost_delta_agents import benchmark_multi_cost_agents

from gridworld.core.logx import setup_logger_kwargs
from gridworld.core.utils import default_termination


# hyper-parameters to test for goes over here
AGENT_LIST = ["s-opt", "h-opt",]

IS_SCHEMES = ["doubly_robust"]
LOWER_BOUND_SCHEMES = ["student_t_test"]

# Dataset generation hyper-parameters
NB_TRAJ_LIST = [10, 50, 500, 2000]
RATIO_LIST  = [0.1, 0.4, 0.7, 0.9]
NB_COSTS_LIST = [1, 4, 16, 64,]

# SPIBB specific parameters
EPS_LIST = [1e-4, 0.001, 0.01, 0.1, 1.0]

DELTA_LIST = [0.1]

# only try for one combination (1,0)
LAMBDA_COEFFS = [[1.0, 0.0]]


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    # env specific arguments
    parser.add_argument('--env_name', type=str, default='costly_large_grid-200')
    parser.add_argument('--cost_lim', type=float, default=10.0)
    parser.add_argument('--env_tries', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    # exp logging specific args
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='scale_comparison')
    parser.add_argument('--out_dir', type=str, default='/tmp/batch-scale/')
    parser.add_argument('--num_runs', type=int, default=1)
    # parse args
    args = parser.parse_args()


    # Create the agents
    exp_agents = []
    for agent_type in AGENT_LIST:

        # Prepare agent
        if agent_type == 's-opt':
            # add the SOP agent to the mix
            agent_kwargs = dict(termination_condition=default_termination,
                                coeff_list=LAMBDA_COEFFS)
            agent = MultiConstSPIBBAgent(**agent_kwargs)
            exp_agents.append(agent)
        elif agent_type == 'h-opt':
            # add different HOPT agents
            for estimator_type in IS_SCHEMES:
                for lower_bound_stratagem in LOWER_BOUND_SCHEMES:
                    # add the agent with IS and LB combination in the list of agents to test
                    agent_kwargs = dict(lower_bound_strategy=lower_bound_stratagem,
                                        estimator_type=estimator_type,
                                        coeff_list=LAMBDA_COEFFS,
                                        training_size=0.7,
                                        )
                    agent = MultiCostHCPIAgent(**agent_kwargs)
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
    benchmark_multi_cost_agents(discount=args.gamma,
                                cost_limit=args.cost_lim,
                                # Exp params
                                num_runs=args.num_runs,
                                num_creation_tries=args.env_tries,
                                agent_list=exp_agents,
                                seed=args.seed,
                                # Experience collection:
                                num_constraints_list=NB_COSTS_LIST,
                                nb_trajectories_list=NB_TRAJ_LIST,
                                ratio_list=RATIO_LIST,
                                # agent specific
                                epsilon_list=EPS_LIST,
                                delta_list=DELTA_LIST,
                                # PI_limit
                                max_PI_limit=1,
                                # Optimization params
                                # Logging:
                                logger_kwargs=logger_kwargs,
                                )

    print(f"Experiment finished")