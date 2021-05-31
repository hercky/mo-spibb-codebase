from gridworld.envs.safety_gridworld import  PitWorld

def create_env(env_name,
               goal_reward=1000.0,
               obstacle_cost=1.0,
               step_penalty=-1.0,
               obstacle_density=0.3,):
    """
    the main method which parses the name and creates the pit-world environment

    we used large_grid environment for our experiments
    """
    env = None

    # find the kind of grid to build
    env_type = env_name.split("-")[0]

    # default episode_len is 50, from SPIBB code
    episode_len = int(env_name.split("-")[1]) if len(env_name.split("-")) > 1 else 50

    # OPE is clipped for max 200 steps in the code
    assert episode_len <= 200

    if env_type == "small_grid":
        # create 5x5 grid with the following params
        env = PitWorld(size=5+2,
                       max_step=episode_len,
                       per_step_penalty=step_penalty,
                       goal_reward=goal_reward,
                       obstace_density=obstacle_density,
                       constraint_cost=obstacle_cost,
                       feature_type="tabular",
                       rand_goal=False, #SPIBB experiments goal is fixed
                       rand_transition=True,
                       random_action_prob=0.0,
                       )
    elif  env_type == "random_small_grid":
        # create 5x5 grid with the following params
        env = PitWorld(size=5+2,
                       max_step=episode_len,
                       per_step_penalty=step_penalty,
                       goal_reward=goal_reward,
                       obstace_density=obstacle_density,
                       constraint_cost=obstacle_cost,
                       feature_type="tabular",
                       rand_goal=True,
                       rand_transition=True,
                       random_action_prob=0.0,
                       )
    elif env_type == "large_grid":
        # create 10x10 grid with the following params
        env = PitWorld(size=10 + 2,
                       max_step=episode_len,
                       per_step_penalty=step_penalty,
                       goal_reward=goal_reward,
                       obstace_density=obstacle_density,
                       constraint_cost=obstacle_cost,
                       feature_type="tabular",
                       rand_goal=False,
                       rand_transition=True,
                       random_action_prob=0.0,
                       )

    else:
        raise Exception("undefined environment")

    return env
