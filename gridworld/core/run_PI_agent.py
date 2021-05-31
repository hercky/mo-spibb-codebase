import numpy as np
import cvxpy as cp

from gridworld.core.utils import *
from gridworld.envs.creation_utils import create_env
from gridworld.core.logx import Logger, setup_logger_kwargs
import pandas as pd

from gridworld.agents.tabular_spibb_agents import ConstSPIBBAgent
from gridworld.agents.tabular_hcpi_agents import HCPIAgent
from gridworld.agents.tabular_base_agent import UnconstPIAgent, RewardShapingPIAgent, ReshapedAdvantagePIAgent

from tqdm import tqdm

import pickle

# Multi-purpose agent runner for policy improvement algos
def run_tabular_batch_PI_agents(env_name,
                                num_creation_tries=1000,
                                agent_list=[],
                                seed=0,
                                # Experience collection:
                                nb_trajectories_list=[],
                                ratio_list= [],
                                # spibb hyper-params
                                epsilon_list=[],
                                delta=1,
                                # hcpi params
                                delta_hcpi_list = [],
                                # MDP params
                                discount=0.99,
                                cost_limit=2.0,
                                # Optimization params
                                max_PI_limit = 50,
                                # Logging:
                                logger=None,
                                logger_kwargs=dict(),
                     ):
    """
    :param env_name:
    :param agent_list:
    :param seed:
    :param nb_trajectories_list:
    :param ratio_list:
    :param epsilon_list:
    :param delta:
    :param discount:
    :param cost_limit:
    :param logger:
    :param logger_kwargs:
    :return:
    """
    # =========================================================================#
    #  Prepare logger, seed, and result store for this run                     #
    # =========================================================================#

    logger = Logger(**logger_kwargs) if logger is None else logger
    # logger also creates the output dir for storing things
    # Note: logger_kwargs contains
    #   - output_dir
    #   - exp_name
    #   - output_filename
    #   - print on std output or only in logs
    # save the experiment variables in a config file
    logger.save_config(locals())

    # set the seed for numpy
    np.random.seed(seed)

    # container to store the results
    results = []

    # =========================================================================#
    #  Creates and validates the environment                                   #
    # =========================================================================#
    i = 0

    # try 1k random envs
    while i < num_creation_tries:
        try:
            logger.log(f"Try Env #: {i}")
            i += 1

            env = create_env(env_name)
            P_star, R_star, C_star, initial_distribution = env.compute_cmdp_matrices()

            #  Compute \pi_* using the dual formulation
            v_opt, c_opt, pi_opt = cmdp_dual_lp(P_star, R_star, C_star, discount, cost_limit, initial_distribution)

            if np.isnan(pi_opt).any():
                # skip the current/wrong solution
                # continue
                raise Exception("Nans in the pi_opt")

            # if able to compute the solution successfully then break
            break
        except Exception:
            # If there is no optimal policy, try with a new environment
            logger.log(f"Couldn't find optimal, trying again.")

    if i >= num_creation_tries:
        raise Exception("Tried 1k environments but can't find \
                        the optimal policy in any of them. \
                        Try again with a simpler environment!")

    # Print the env to logs for visualisation later
    logger.log(env.to_string())

    # save the env
    with open(f'{logger.output_dir}/env.p', 'wb') as f:
            pickle.dump(env, f)

    # get the |S| and |A|
    nstates = env.nstates
    nactions = env.nactions

    # create a uniform random policy for mixing with the ratio specified with
    pi_random = np.ones((nstates, nactions)) / nactions

    # calculate the max and min return values
    r_min = env.max_step * env.per_step_penalty
    r_max = env.goal_reward
    c_min = 0
    c_max = env.max_step * env.constraint_cost


    # save these stats (they'll come handy in plotting later)
    with open(f'{logger.output_dir}/env_stats.p', 'wb') as f:
        pickle.dump([r_min, r_max, c_min, c_max], f)


    # =========================================================================#
    #  Compute different agents and parameter for the same random MDP grid
    # =========================================================================#
    for ratio in tqdm(ratio_list, desc="Main Loop (ratios)"):

        logger.log("--rho--")
        print(f"ratio: {ratio}")
        # =====================================================================#
        #  Corrupt the policy using the ratio
        # =====================================================================#
        pi_baseline = ratio * pi_opt + (1-ratio) * pi_random
        # make sure no method can change the pi_baseline (make it read-only)
        pi_baseline.flags.writeable = False

        for nb_traj in tqdm(nb_trajectories_list, desc="Nested Loop (nb_traj)"):
            # =========================================================================#
            #  Gather data under \pi_b                                                 #
            # =========================================================================#
            trajectories, batch_transitions = generate_dataset(nb_traj, env, pi_baseline)

            # =========================================================================#
            #  Estimate the MLE estimates and the error bounds
            # =========================================================================#

            # As with SPIBB/Soft-SPIBB we are using the true reward and
            # cost model as they are not stochastic in this case
            # When they are stochastic, they should be estimated also
            P_hat = estimate_model(batch_transitions, nstates, nactions)
            eQ = compute_error_function(batch_transitions, nstates, nactions, delta)

            # =========================================================================#
            #  Benchmark baseline
            #  Use the direct policy evaluation methods to get the performance
            # =========================================================================#
            logger.log("--- Benchmarking baseline ---")
            # w.r.t R
            vR_pib_mhat = direct_policy_evaluation(P_hat, R_star, discount, pi_baseline)
            pib_R_est_performance = sum(vR_pib_mhat * initial_distribution)
            # print(f"V^(\pib)_(Mhat)(R) {pib_R_est_performance}")
            logger.log(f"V^(pib)_(Mhat)(R) {pib_R_est_performance}")
            # w.r.t. C
            vC_pib_mhat = direct_policy_evaluation(P_hat, C_star, discount, pi_baseline)
            pib_C_est_performance = sum(vC_pib_mhat * initial_distribution)
            logger.log(f"V^(pib)_(Mhat)(C)  {pib_C_est_performance}")

            # compute performance w.r.t to the true M*
            # V^{\pib})_{M*}(R)
            vR_pib_mopt = direct_policy_evaluation(P_star, R_star, discount, pi_baseline)
            pib_R_true_performance = sum(vR_pib_mopt * initial_distribution)
            logger.log(f"V^(pib))_(M*)(R) {pib_R_true_performance}")
            # V^{\pib})_{M*}(C)
            vC_pib_mopt = direct_policy_evaluation(P_star, C_star, discount, pi_baseline)
            pib_C_true_performance = sum(vC_pib_mopt * initial_distribution)
            logger.log(f"V^(pib))_(M*)(C) {pib_C_true_performance}")

            # =========================================================================#
            #  Benchmark different agent and hyper-params for the same env and dataset
            # =========================================================================#
            for agent in agent_list:

                # if the agent has lambda_coeff list iterate over them, else do only one run
                for coeff in agent.coeff_list or [None]:

                    if "SPIBB" in agent.__class__.__name__:
                        # if a SPIBB based agent

                        for epsilon in epsilon_list:
                            # =========================================================================#
                            #  Use the agent's PI update algorithm to do Policy Improvement
                            #   for each epsilon
                            # =========================================================================#

                            # make the operator with current parameters
                            agent_operator = agent.make_policy_iteration_operator(P=P_hat, R=R_star,
                                                                                  C=C_star, discount=discount,
                                                                                  baseline=pi_baseline,
                                                                                  error_fn=eQ,
                                                                                  epsilon=epsilon,
                                                                                  coeffs=coeff,
                                                                                  )

                            # successive approximation
                            try:
                                # Note: we are giving the baseline policy as the initial policy to the operator
                                #        this can be the random policy also i.e. pi_random
                                pi_solution = bounded_successive_approximation(pi_baseline,
                                                                               operator=agent_operator,
                                                                               termination_condition=agent.termination_condition,
                                                                               max_limit=max_PI_limit, )
                            except cp.error.SolverError:
                                # if unable to solve return the baseline
                                logger.log("Couldn't solve, returning baseline")
                                pi_solution = pi_baseline

                            # log performance on the true and estimated models
                            logger.log("--- Benchmarking solution for ")
                            logger.log(f"Ratio:{ratio}\t Num_traj:{nb_traj}\t Agent:{agent._name}\t Eps:{epsilon}\t Coeff:{coeff}")

                            # w.r.t. P_hat
                            vR_piSolution_mhat = direct_policy_evaluation(P_hat, R_star, discount, pi_solution)
                            piSolution_R_est_performance = sum(vR_piSolution_mhat * initial_distribution)
                            logger.log(f"V^(pi_SOL)_(Mhat)(R) {piSolution_R_est_performance}")
                            vC_piSolution_mhat = direct_policy_evaluation(P_hat, C_star, discount, pi_solution)
                            piSolution_C_est_performance = sum(vC_piSolution_mhat * initial_distribution)
                            logger.log(f"V^(pi_SOL))_(Mhat)(C) {piSolution_C_est_performance}")

                            # w.r.t. P_star
                            vR_piSolution_mopt = direct_policy_evaluation(P_star, R_star, discount, pi_solution)
                            piSolution_R_true_performance = sum(vR_piSolution_mopt * initial_distribution)
                            logger.log(f"V^(pi_SOL))_(M*)(R) {piSolution_R_true_performance}")
                            vC_piSolution_mopt = direct_policy_evaluation(P_star, C_star, discount, pi_solution)
                            piSolution_C_true_performance = sum(vC_piSolution_mopt * initial_distribution)
                            logger.log(f"V^(pi_SOL))_(M*)(C) {piSolution_C_true_performance}")

                            # =========================================================================#
                            #  Save the results and Log
                            # =========================================================================#

                            # save all the statistics
                            results.append([seed,                                    # exp run
                                            discount, cost_limit, nstates, nactions, # MDP params
                                            nb_traj, ratio,                          # baseline params
                                            pib_R_true_performance, pib_C_true_performance, # baseline's true performance
                                            pib_R_est_performance, pib_C_est_performance,  # baseline's estimated performance
                                            piSolution_R_true_performance, piSolution_C_true_performance, # sol true perf
                                            piSolution_R_est_performance, piSolution_C_est_performance, # sol estim perf
                                            agent._name, # Agent name/kind
                                            "_".join(map(str, coeff)),  # conver coeff to string
                                            delta, epsilon,                 # spibb specific params
                                            None,  None,  None ,                        # hcpi specific params
                                            ])

                    elif "HCPI" in agent.__class__.__name__:
                        # =========================================================================#
                        #  Do HCPI for different \delta parameter
                        # =========================================================================#

                        for delta_hcpi in delta_hcpi_list:

                            # use the agent's param to get the solution
                            pi_solution, reg_coeff = agent.compute_policy(trajectories=trajectories,
                                                                          pi_b=pi_baseline,
                                                                          confidence=delta_hcpi/2., # because of union bound
                                                                          coeffs=coeff,
                                                                          R=R_star,
                                                                          C=C_star,
                                                                          discount=discount,
                                                                          pib_R_est_performance=pib_R_est_performance,
                                                                          pib_C_est_performance=pib_C_est_performance,
                                                                          R_min=r_min,
                                                                          R_max=r_max,
                                                                          C_min=c_min,
                                                                          C_max=c_max,
                                                                          q_learning_iterative_mode=False,
                                                                          )

                            # log performance on the true and estimated models
                            logger.log("--- Benchmarking solution for ")
                            logger.log(f"Ratio:{ratio}\t Num_traj:{nb_traj}\t Agent:{agent._name}\t Coeff:{coeff}")
                            logger.log(f"Delta:{delta_hcpi}\t OPE Estimator type:{agent.estimator_type}")


                            # w.r.t. P_hat
                            vR_piSolution_mhat = direct_policy_evaluation(P_hat, R_star, discount, pi_solution)
                            piSolution_R_est_performance = sum(vR_piSolution_mhat * initial_distribution)
                            logger.log(f"V^(pi_SOL)_(Mhat)(R) {piSolution_R_est_performance}")
                            vC_piSolution_mhat = direct_policy_evaluation(P_hat, C_star, discount, pi_solution)
                            piSolution_C_est_performance = sum(vC_piSolution_mhat * initial_distribution)
                            logger.log(f"V^(pi_SOL))_(Mhat)(C) {piSolution_C_est_performance}")

                            # w.r.t. P_star
                            vR_piSolution_mopt = direct_policy_evaluation(P_star, R_star, discount, pi_solution)
                            piSolution_R_true_performance = sum(vR_piSolution_mopt * initial_distribution)
                            logger.log(f"V^(pi_SOL))_(M*)(R) {piSolution_R_true_performance}")
                            vC_piSolution_mopt = direct_policy_evaluation(P_star, C_star, discount, pi_solution)
                            piSolution_C_true_performance = sum(vC_piSolution_mopt * initial_distribution)
                            logger.log(f"V^(pi_SOL))_(M*)(C) {piSolution_C_true_performance}")

                            # =========================================================================#
                            #  Save the results and Log
                            # =========================================================================#

                            # for compatibility with Soft-SPIBB code
                            results.append([seed,  # exp run
                                            discount, cost_limit, nstates, nactions,  # MDP params
                                            nb_traj, ratio,  # baseline params
                                            pib_R_true_performance, pib_C_true_performance,  # true performance
                                            pib_R_est_performance, pib_C_est_performance,  # baselines estimated performance
                                            piSolution_R_true_performance, piSolution_C_true_performance,  # sol true perf
                                            piSolution_R_est_performance, piSolution_C_est_performance,  # sol estim perf
                                            agent._name,  # agent
                                            "_".join(map(str,coeff)),  # conver coeff to string
                                            None, None,  # spibb specific params
                                            delta_hcpi, agent.estimator_type, agent.lower_bound_strategy,            # hcpi specific params
                                            ])

                    else:
                        # =========================================================================#
                        #  It is one of the baseline agents: Unconst, RewardShaping, RewardShaping w/ Advantage
                        # =========================================================================#

                        # make the operator with current parameters
                        agent_operator = agent.make_policy_iteration_operator(P=P_hat, R=R_star,
                                                                              C=C_star, discount=discount,
                                                                              baseline=pi_baseline,
                                                                              coeffs=coeff,
                                                                              )

                        # successive approximation
                        try:
                            # We are giving the baseline policy as the initial policy to the operator
                            # but this can be the random policy as well i.e. pi_random
                            pi_solution = bounded_successive_approximation(pi_baseline,
                                                                           operator=agent_operator,
                                                                           termination_condition=agent.termination_condition,
                                                                           max_limit=max_PI_limit, )
                        except cp.error.SolverError:
                            # if unable to solve return the baseline
                            logger.log("Couldn't solve, returning baseline")
                            pi_solution = pi_baseline

                        # log performance on the true and estimated models
                        logger.log("--- Benchmarking solution for ")
                        logger.log(
                            f"Ratio:{ratio}\t Num_traj:{nb_traj}\t Agent:{agent._name}\t Coeff:{coeff}")

                        # w.r.t. P_hat
                        vR_piSolution_mhat = direct_policy_evaluation(P_hat, R_star, discount, pi_solution)
                        piSolution_R_est_performance = sum(vR_piSolution_mhat * initial_distribution)
                        logger.log(f"V^(pi_SOL)_(Mhat)(R) {piSolution_R_est_performance}")
                        vC_piSolution_mhat = direct_policy_evaluation(P_hat, C_star, discount, pi_solution)
                        piSolution_C_est_performance = sum(vC_piSolution_mhat * initial_distribution)
                        logger.log(f"V^(pi_SOL))_(Mhat)(C) {piSolution_C_est_performance}")

                        # w.r.t. P_star
                        vR_piSolution_mopt = direct_policy_evaluation(P_star, R_star, discount, pi_solution)
                        piSolution_R_true_performance = sum(vR_piSolution_mopt * initial_distribution)
                        logger.log(f"V^(pi_SOL))_(M*)(R) {piSolution_R_true_performance}")
                        vC_piSolution_mopt = direct_policy_evaluation(P_star, C_star, discount, pi_solution)
                        piSolution_C_true_performance = sum(vC_piSolution_mopt * initial_distribution)
                        logger.log(f"V^(pi_SOL))_(M*)(C) {piSolution_C_true_performance}")

                        # =========================================================================#
                        #  Save the results and Log
                        # =========================================================================#

                        # save all the statistics
                        results.append([seed,  # exp run
                                        discount, cost_limit, nstates, nactions,  # MDP params
                                        nb_traj, ratio,  # baseline params
                                        pib_R_true_performance, pib_C_true_performance,  # baseline's true performance
                                        pib_R_est_performance, pib_C_est_performance,
                                        # baseline's estimated performance
                                        piSolution_R_true_performance, piSolution_C_true_performance,  # sol true perf
                                        piSolution_R_est_performance, piSolution_C_est_performance,  # sol estim perf
                                        agent._name,  # Agent name/kind
                                        "_".join(map(str, coeff)),  # conver coeff to string
                                        None, None,  # spibb specific params
                                        None, None, None,  # hcpi specific params
                                        ])



        # All experiments are finished, save the results

        # Dump the results in a dataframe similar to SPIBB
        df = pd.DataFrame(results, columns=['seed',
                                            'gamma', 'cost_limit', 'nb_states', 'nb_actions',
                                            'nb_trajectories', 'ratio',
                                            'pib_R_true_performance', 'pib_C_true_performance',
                                            'pib_R_est_performance', 'pib_C_est_performance',
                                            'piSolution_R_true_performance', 'piSolution_C_true_performance',
                                            'piSolution_R_est_performance', 'piSolution_C_est_performance',
                                            'agent_name',
                                            'coeff',
                                            'delta', 'epsilon',
                                            'delta_hcpi', 'IS_estimator', 'lower_bound_strategy',
                                            ])

        # Save the files here
        logger.dump_df_as_xlsx(df)
        df.to_csv(path_or_buf=logger.result_file + ".csv")

        logger.log(f"{len(results)} lines saved to {logger.result_file} in .xlsx and .csv")



if __name__ == '__main__':
    """
    test for a single agent and hyper-param combination here 
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='small_grid-25')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cost_lim', type=float, default=2.0)
    parser.add_argument('--agent', type=str, default="h-opt")
    parser.add_argument('--exp_name', type=str, default='runagent')
    parser.add_argument('--nb_traj', type=int, default=200)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--ope', type=str, default="importance_sampling")

    # parse args
    args = parser.parse_args()

    # Prepare logger
    from gridworld.core.logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name,
                                        env_name=args.env_name,
                                        seed=args.seed,
                                        data_dir="/tmp/mo-spibb/",
                                        print_along=True)

    # Prepare coefficients list
    lambda_R_vals = [1.0]  # >=0
    lambda_C_vals = [1.0]  # >=0
    lambda_coeffs = [(lr, lc) for lc in lambda_C_vals for lr in lambda_R_vals]


    # Prepare agent
    if args.agent == 's-opt':
        agent_kwargs = dict(termination_condition=default_termination,
                            coeff_list=lambda_coeffs)

        agent = ConstSPIBBAgent(**agent_kwargs)
    elif args.agent == 'h-opt':

        agent_kwargs = dict(lower_bound_strategy="student_t_test",
                            estimator_type=args.ope,
                            coeff_list=lambda_coeffs,
                            training_size=0.7,
                            )

        agent = HCPIAgent(**agent_kwargs)
    elif args.agent == 'reg-PI':
        agent_kwargs = dict(termination_condition=default_termination,)

        agent = UnconstPIAgent(**agent_kwargs)
    elif args.agent == 'rs-PI':
        agent_kwargs = dict(termination_condition=default_termination,
                            coeff_list=lambda_coeffs)

        agent = RewardShapingPIAgent(**agent_kwargs)
    elif args.agent == 'adv-rs-PI':
        agent_kwargs = dict(termination_condition=default_termination,
                            coeff_list=lambda_coeffs)

        agent = ReshapedAdvantagePIAgent(**agent_kwargs)
    else:
        raise Exception("not implemented yet")


    run_tabular_batch_PI_agents(args.env_name,
                                num_creation_tries=10,
                                agent_list=[agent],
                                seed=args.seed,
                                # Experience collection:
                                nb_trajectories_list=[args.nb_traj],
                                ratio_list= [0.0, args.ratio],
                                # agent specific
                                epsilon_list=[args.eps],
                                delta_hcpi_list=[0.9],
                                # PI_limit
                                max_PI_limit=10,
                                # MDP params
                                discount=args.gamma,
                                cost_limit=args.cost_lim,
                                # Optimization params
                                # Logging:
                                logger_kwargs=logger_kwargs,
                                )