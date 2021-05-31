"""
Created the CMDP with pits and goals in Sec Synthetic Experiments

Main code from here: https://github.com/junhyukoh/value-prediction-network/blob/master/maze.py
And visualization inspired from: https://github.com/WojciechMormul/rl-grid-world
"""


from PIL import Image
import numpy as np
import gym
from gym import spaces
import copy

# constants
BLOCK = 0
AGENT = 1
GOAL = 2
PIT = 3

# movemnent, can only move in 4 directions
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]


# for generation purposes
COLOR = [
        [44, 42, 60], # block - black
        [91, 255, 123], # agent - green
        [52, 152, 219], # goal - blue
        [255, 0, 0], # pit - red
        ]



def generate_maze(size=27, obstacle_density=0.3, rand_goal=True):
    """
    Generate a CMDP maze layout with random goal state (in the leftmost column), random pits placement
     based on the obstacle density ratio

     The start state is always fixed in a corner

    :param size: (int) the size (n) of the square maze (n x n)
    :param obstacle_density: (float) the uniform prob of placing a pit at any cell
    :param rand_goal: (bool) if the goal is random or not

    :return: the maze layout (np.ndarray), start coordinates (tuple (int,int)), goal coordinates (tuple (int, int))

    """
    mx = size-2; my = size-2 # width and height of the maze
    maze = np.zeros((my, mx))

    #NOTE: padding here
    dx = DX
    dy = DY


    # define the start and the goal
    # start in the corner
    start_y, start_x =  my-1, mx-1

    # goal position   (24,24)
    if rand_goal:
        goal_y, goal_x = np.random.randint(0,my), 0
    else:
        # goal_y, goal_x = my//2, 0
        goal_y, goal_x = 0, 0


    # create the actual maze here
    # maze_tensor = np.zeros((size, size, len(COLOR)))
    maze_tensor = np.zeros((len(COLOR), size, size))

    # fill everything with blocks
    # maze_tensor[:,:,BLOCK] = 1.0
    maze_tensor[BLOCK,:,:] = 1.0

    # fit the generated maze
    # maze_tensor[1:-1, 1:-1, BLOCK] = maze
    maze_tensor[BLOCK, 1:-1, 1:-1] = maze

    # put the agent
    # maze_tensor[start_y+1][start_x+1][AGENT] = 1.0
    maze_tensor[AGENT][start_y+1][start_x+1]= 1.0

    # put the goal
    # maze_tensor[goal_y+1][goal_x+1][GOAL] = 1.0
    maze_tensor[GOAL][goal_y+1][goal_x+1] = 1.0


    # put the pits

    # create the the pits here
    for i in range(0, mx):
        for j in range(0, my):
            # pass if start or goal state
            if (i==start_x and j==start_y) or (i==goal_x and j==goal_y):
                pass

            # with prob p place the pit
            if np.random.rand() < obstacle_density:
                # maze_tensor[j+1][i+1][PIT] = 1.0
                maze_tensor[PIT][j+1][i+1] = 1.0


    return maze_tensor, [start_y+1, start_x+1], [goal_y+1, goal_x+1]




class PitWorld(gym.Env):
    """
    The gym environment corresponding to the maze experiments with pits
    """
    def __init__(self,
                 size = 25,
                 max_step = 200,
                 per_step_penalty = -1.0,
                 goal_reward = 1000.0,
                 obstace_density = 0.3,
                 constraint_cost = 1.0,
                 random_action_prob = 0.000,  #0.005
                 rand_goal = True,
                 rand_transition = False,
                 feature_type="tabular"):
        """
        Initializes the Pit world environment

        :param size: (int) the size of the maze
        :param max_step: (int ) horizon after which the episode ends
        :param per_step_penalty: (float) the per step penalty the agent gets for taking any action
        :param goal_reward: (float) the reward on reaching the goal state
        :param obstace_density: (float) the probability of randomly spawning a pit on the maze cell
        :param constraint_cost: (float) the cost the agent gets on stepping into the pit cell
        :param random_action_prob: (float) the prob with which a random action is taken (to make the env stochastic)
        :param rand_goal: (bool) whether the goal should be randomly generated or fixed
        :param rand_transition: (bool) whether a stochastic transition matrix is used or not
                                    If this is true, then it overrised random_action_prob
        :param feature_type: (str) if the feature type is:
                    - one-hot: one hot vector specifying the location of the agent
                    - tensor: the full ndarray of the pit is returned
                    - tabular: a unique integer correspoding to the grid number
        """

        self.size = size
        self.dy = DY
        self.dx = DX
        self.random_action_prob = random_action_prob
        self.per_step_penalty = per_step_penalty
        self.goal_reward = goal_reward
        self.obstace_density = obstace_density
        self.max_step = max_step
        self.constraint_cost = constraint_cost
        self.feature_type = feature_type
        self.rand_goal = rand_goal
        self.rand_transition = rand_transition


        # because there are walls on periphery of the grid
        self.nstates = (size-2) * (size-2)

        # 4 possible actions
        self.nactions = 4
        self.action_space = spaces.Discrete(self.nactions)

        # create the maze
        self.init_maze, self.start_pos, self.goal_pos = generate_maze(size=self.size,
                                                                      obstacle_density=self.obstace_density,
                                                                      rand_goal=self.rand_goal)


        # observation space
        # TODO: 4d tensor or 3d image
        if self.feature_type == "tensor":
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=self.init_maze.shape)
        elif self.feature_type == "one-hot":
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=self.init_maze[AGENT].shape)
        elif self.feature_type == "tabular":
            # (n x n) unique cells
            self.observation_space = spaces.Discrete(self.nstates)
        else:
            raise Exception("not implemented yet!")

        if self.random_action_prob > 0 and self.rand_transition:
            raise Exception("can't have both modes have stochasticity, either have stocahstic transition functinos or sticky actions")

        # define the episode specific variables
        self.maze = copy.deepcopy(self.init_maze)
        self.agent_pos = copy.deepcopy(self.start_pos)

        self.t = 0
        self.episode_reward = 0
        self.done = False



        # if random transition matrix generate so here
        self.transition_function = np.zeros((self.nstates, self.nactions, self.nstates))
        if rand_transition:
            self.generate_random_transition()


    def reset(self):
        """
        same as gym's reset function

        :return: returns the initial state
        """
        self.maze = copy.deepcopy(self.init_maze)
        self.agent_pos = copy.deepcopy(self.start_pos)

        self.t = 0
        self.episode_reward = 0
        self.done = False

        return self.observation()


    def cord_to_index(self, y, x):
        """
        converts the cordinates (y,x) to a unique identifier (int)

        :param y:
        :param x:
        :return: (int) the index
        """
        return (y-1) * (self.size-2) + (x-1)


    def observation(self):
        obs = np.array(self.maze, copy=True)

        if self.feature_type == "tensor":
            # returns in the (channel, height, width) format
            obs = np.reshape(obs, (-1, self.size, self.size))
        elif self.feature_type == "one-hot":
            obs = obs[AGENT].flatten()
        elif self.feature_type == "tabular":
            obs = self.cord_to_index(self.agent_pos[0], self.agent_pos[1])
            # alternate is to create the map from (pos, pos) -> int
        else:
            raise Exception("not implemented yet!")

        return obs

    def generate_random_transition(self):
        """
        the way this works is that it, if the agent takes an action 'a' then:
        with prob: p ~ U(0,1] gives a random prob to giving in that direction
        and ()

        TODO: the episode always ends on the goal, do we still need to estimate that?
        """
        my = self.maze.shape[1]
        mx = self.maze.shape[2]

        for y in range(1, my-1):
            for x in range(1, mx-1):
                # state is (y,x)
                current_state = self.cord_to_index(y,x)

                for action in range(self.nactions):
                    # calculate the next_state for an action

                    ny = y + DY[action]
                    nx = x + DX[action]

                    next_state = self.cord_to_index(ny, nx)

                    prob_next_state = np.random.uniform(0.5, 1.0)
                    stay_prob = 1.0 - prob_next_state

                    if self.is_reachable(ny, nx):
                        self.transition_function[current_state, action, next_state] = prob_next_state
                        self.transition_function[current_state, action, current_state] = stay_prob
                    else:
                        #stay there
                        self.transition_function[current_state, action, current_state] = 1.0

                    #done with this action

        # done with all states


    def compute_cmdp_matrices(self):
        """
        compute and return the transition, reward and cost matrix, and initial state dist
         for the current env

        :return: P,R,C,Mu
        """

        # compute the transition function
        P = self.transition_function.copy()
        terminal_state= self.cord_to_index(self.goal_pos[0], self.goal_pos[1])
        P[terminal_state, :, :] = 0.0

        # compute the reward function
        # have the per-step penalty function here
        R = np.ones((self.nstates, self.nactions)) * self.per_step_penalty
        # set the reward for goal state to goal reward
        R[terminal_state, :] = self.goal_reward

        # get the cost function for pits here
        C = np.zeros((self.nstates, self.nactions))

        my = self.maze.shape[1]
        mx = self.maze.shape[2]

        for y in range(1, my - 1):
            for x in range(1, mx - 1):
                # state is (y,x)
                current_state = self.cord_to_index(y, x)
                # state beased constraints
                if self.maze[PIT][y][x] == 1.0:
                    C[current_state, :] = self.constraint_cost

        # fixed starting position in this case
        Mu = np.zeros(self.nstates)
        Mu[self.cord_to_index(self.start_pos[0], self.start_pos[1])] = 1.0

        return P, R, C, Mu

    def visualize(self, img_size=320):
        """
        FIXME: fix and verify this part later
        """
        # return an exception for now
        raise Exception("not fixed yet :(")

        img_maze = np.array(self.maze, copy=True).reshape(self.size, self.size, -1)
        #         currently for maze[y][x][color]
        my = self.maze.shape[0]
        mx = self.maze.shape[1]
        colors = np.array(COLOR, np.uint8)
        num_channel = self.maze.shape[2]
        vis_maze = np.matmul(self.maze, colors[:num_channel])
        vis_maze = vis_maze.astype(np.uint8)
        for i in range(vis_maze.shape[0]):
            for j in range(vis_maze.shape[1]):
                if self.maze[i][j].sum() == 0.0:
                    vis_maze[i][j][:] = int(255)
        image = Image.fromarray(vis_maze)
        return image.resize((int(float(img_size) * mx / my), img_size), Image.NEAREST)

    def to_string(self):
        """
        :return: (str) an ASCII version of the maze
        """
        my = self.maze.shape[1]
        mx = self.maze.shape[2]
        str = ''
        for y in range(my):
            for x in range(mx):
                if self.maze[BLOCK][y][x] == 1:
                    str += '#'
                elif self.maze[AGENT][y][x] == 1:
                    str += 'A'
                elif self.maze[GOAL][y][x] == 1:
                    str += 'G'
                elif self.maze[PIT][y][x] == 1:
                    str += 'x'
                else:
                    str += ' '
            str += '\n'
        return str

    def is_reachable(self, y, x):
        """
        checks if the agent can reach the position at (y,x)
        :param y: int
        :param x: int
        :return: bool
        """
        # if there is no block
        return self.maze[BLOCK][y][x] == 0


    def move_agent(self, direction):
        """
        Moves the agent in a direction

        :param direction: (int) between [0-3]
        :return: (bool) if the move action was successful
        """
        y = self.agent_pos[0] + self.dy[direction]
        x = self.agent_pos[1] + self.dx[direction]

        if not self.is_reachable(y, x):
            return False

        # else move the agent
        self.maze[AGENT][self.agent_pos[0]][self.agent_pos[1]] = 0.0
        self.maze[AGENT][y][x] = 1.0
        self.agent_pos = [y, x]

        # moved the agent
        return True

    def stochastic_move_agent(self, action):
        """
        Moves the agent in a direction

        :param direction: (int) between [0-3]
        :return: (bool) if the move action was successful
        """
        # for stochasticity, overwrite random action
        if self.random_action_prob > 0:
            # randomize the action
            if np.random.rand() < self.random_action_prob:
                action = np.random.choice(range(len(DX)))
            self.move_agent(action)
        elif self.rand_transition:
            current_state = self.cord_to_index(self.agent_pos[0], self.agent_pos[1])

            y = self.agent_pos[0] + self.dy[action]
            x = self.agent_pos[1] + self.dx[action]

            next_state = self.cord_to_index(y, x)

            # move according to the transition matrix
            if self.is_reachable(y,x) and np.random.rand() < self.transition_function[current_state, action, next_state]:
                self.move_agent(action)
        else:
            raise Exception("No source of randomness, shouldn't move.")


    def step(self, action):
        """
        executes an action in the environment

        :param action: the action to take
        :return:
        """
        assert self.action_space.contains(action)

        constraint = 0
        info = {}

        # increment
        self.t += 1

        # for stochasticity, overwrite random action
        self.stochastic_move_agent(action)

        # default reward
        reward = self.per_step_penalty

        # if reached GOAL
        if self.maze[GOAL][self.agent_pos[0]][self.agent_pos[1]] == 1.0:
            reward = self.goal_reward
            self.done = True

        # if reached PIT
        if self.maze[PIT][self.agent_pos[0]][self.agent_pos[1]] == 1.0:
            constraint = self.constraint_cost

        # if max time steps reached
        if self.t >= self.max_step:
            self.done = True

        if self.t == 1:
            info['begin'] = True
        else:
            info['begin'] = False

        info['pit'] = constraint


        return self.observation(), reward, self.done, info




if __name__ == "__main__":

    env = PitWorld(size=5+2, feature_type="tabular", rand_goal=False, rand_transition=True, random_action_prob=0.0)

    s = env.reset()

    print(env.maze.shape)
    print(env.maze[AGENT].shape)
    print(env.to_string())
    print(env.agent_pos)

    env.compute_cmdp_matrices()
    breakpoint()

    # for a in range(4):
    for _ in range(50):

        print("0->u, 1->r, 2->d, 3->l")
        a = int(input())
        if a not in range(4):
            # go down
            if a == -1:
                s = env.reset()

            a = 0


        # print( DY[a], DX[a])
        s, r, d, info = env.step(a)

        print(s)
        print(env.to_string())
        print(env.agent_pos)
        print(env.t)
        print(env.done)
        print(info)

    print("--------------------------------------------")
    s = env.reset()
    print("--------------------------------------------")
