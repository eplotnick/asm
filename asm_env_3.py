import gym
from gym import spaces
import pdb
import numpy as np
import matplotlib.pyplot as plt
import drawnow
# grid type enums
MINING = 1
# 3: removed farming, replace with empty
EMPTY = 0

DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey eviction
                   # 3: removed farming color
                   'M': [255, 255, 0],  # Yellow mining ground

                   # Colours for agents. R value is a unique identifier
                   '0': [159, 67, 255],  # Purple
                   '1': [2, 81, 154],  # Blue
                   '2': [254, 151, 0],  # Orange
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [100, 255, 255],  # Cyan
                   '6': [99, 99, 255],  # Lavender
                   '7': [250, 204, 255],  # Pink
                   '8': [238, 223, 16]}  # Yellow

def one_hot_to_index(arr):
    if arr.ndim != 1 and arr.shape[1] > 1:
        raise InputError("Must be 1D array or 2D array with 2nd dim as 1")
    arr.astype(int)
    arr_as_list = list(np.reshape(arr, -1))
    return arr_as_list.index(1)

class Government(object):

    def __init__(self, tool_scale = 1.1, scarcity_scale = 0.9,
            merc_efficiency_scale = 1.5, evict_every = 40):
        self._tool_scale  = tool_scale 
        self._scarcity_scale = scarcity_scale
        self._evict_every = evict_every

    @property
    def tool_scale (self):
        return self._tool_scale 

    @property
    def scarcity_scale(self):
        return self._scarcity_scale

    @property
    def merc_efficiency_scale(self):
        return self._scarcity_scale

    @property
    def evict_every(self):
        return self._evict_every

class ASMEnv(gym.Env):

    metadata = {"render.modes": ["human"]}
    def __init__(self, num_agents, govt, episode_length,
            mining_prob_bounds=[0.4, 0.75], 
            beta = 3, is_global_obs=True, reset_mining_probs_every_ep=True):
        super(ASMEnv, self).__init__()

        self._num_agents = num_agents
        self.govt = govt

        # update with the government's policy parameters
        self.evict_every = govt.evict_every
        self.tool_scale = govt.tool_scale
        self.scarcity_scale = govt.scarcity_scale
        self.merc_efficiency_scale = govt.merc_efficiency_scale
        self.episode_length = episode_length
        self.mining_prob_bounds = mining_prob_bounds
        self.is_global_obs = is_global_obs # whether the observation is global
        self.reset_mining_probs_every_ep = reset_mining_probs_every_ep # flips if we reset the map every ep
        self.beta = beta # 3: tweaks contamination curve

        self.width = 5
        self.height = 10
        self.mining_height = 5
        # 3: removed farming_height
        self.obs_width = 7
        self.obs_height = 7
        # 3: NOTE THERE ARE NOW 6 ACTIONS
        # action space is a list of gym Spaces, length num_agents
        self._action_space = [spaces.Discrete(6) for _ in range(self.num_agents)]

        # observations are an array containing, for each agent, their coords
        # and a bool representing whether they've been evicted
        single_agent_obs = spaces.Box(low=0, high=self.height-1,
            shape=(self.num_agents, 3), dtype=np.int32)
        # observation space is a list of gym Spaces, length num_agents
        self._observation_space = [single_agent_obs
            for _ in range(self.num_agents)]

        # up, right, down, left in terms of (x, y) coords
        self.moves = ((0, -1), (1, 0), (0, 1), (-1, 0))

        self.step_count = None
        self.agent_positions = None
        self.world_state = None
        self.evictions = None
        # 3: add this to the observation space? TODO? 
        self.agent_merc_history = None # 3: instead of farming, merc hist
        # this is for the cumulative computation
        self.agent_merc_cumulative = None 
        self.total_merc_actions = None
        self.total_mine_actions = None
        # initialize mining probabilities
        self.reset_mining_probs()

        self.reset()

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, actions):
        # execute one timestep
        self.step_count += 1
        assert(len(actions) == self.num_agents)
        reward = [0 for _ in range(self.num_agents)]
        # perform evictions every evict_every timesteps
        if self.step_count % self.evict_every == 0:
            evicted_agent = self.evict()
            # 3: reset agent_merc_history to 0 after evicting 
            self.agent_merc_history = [0 for _ in range(self.num_agents)]
        else:
            evicted_agent = None
        
        # 3: updates mining probabilities
        self.update_mining_probs()
        # 3: reset prev rounds' mining action sums
        self.world_state[:, :, -3] = 0
        shuffled_agent_indices = np.arange(len(actions))
        # handle agent actions randomly - create randomly ordered index
        np.random.shuffle(shuffled_agent_indices)
        for ind in shuffled_agent_indices:
            if evicted_agent is not None and ind == evicted_agent:
                # don't execute their action if they were just evicted
                continue
            #agent_action = one_hot_to_index(actions[ind])
            agent_action = actions[ind]
            curr_agent_position = self.agent_positions[ind, :]
            if agent_action == 5:
                agent_reward = self.mine(curr_agent_position, ind)
                reward[ind] = agent_reward
            elif agent_action == 4:
                agent_reward = self.mine_and_merc(curr_agent_position, ind)
                reward[ind] = agent_reward
            else:
                x_move, y_move = self.moves[agent_action]
                new_pos = curr_agent_position[0] + x_move, curr_agent_position[1] + y_move
                # check to see the new position is within bounds, and another
                # agent hasn't moved into that position, else stay put
                if (new_pos[0] < self.width and new_pos[0] >= 0) and (new_pos[1] < self.height and new_pos[1] >= 0):
                    if self.world_state[new_pos[0], new_pos[1], -1] < 0:
                        # update agent position
                        self.world_state[new_pos[0], new_pos[1], -1] = ind
                        # remove from old pos
                        self.world_state[curr_agent_position[0], curr_agent_position[1], -1] = -1
                        self.agent_positions[ind, :] = new_pos[0], new_pos[1]
        obs = self.get_observations()
        done = self.step_count >= self.episode_length
        info = [{} for _ in range(self.num_agents)]

        return obs, reward, done, info
   
    def mine(self, coords, agent_id):
        grid_type = self.get_grid_type(coords)
        if grid_type == MINING:
            self.total_mine_actions[agent_id] += 1
            reward = self.get_mining_reward(coords)
            # keep track of mining history if successfully mined
            if reward:
                prev_num_miners = self.world_state[coords[0], coords[1], agent_id]
                self.world_state[coords[0], coords[1], agent_id] = prev_num_miners + 1
                # 3: give tool scale  
                reward *= self.tool_scale
        else:
            reward = 0 
        
        return reward

    def mine_and_merc(self, coords, agent_id):
        grid_type = self.get_grid_type(coords)
        if grid_type == MINING:
            self.total_merc_actions[agent_id] += 1
            reward = self.get_mining_reward(coords)
            # keep track of mining history if successfully mined
            if reward:
                prev_num_miners = self.world_state[coords[0], coords[1], agent_id]
                self.world_state[coords[0], coords[1], agent_id] = prev_num_miners + 1
                # 3: keep track of merc damage to a location and in an agent's history
                # 3: damage cost is the _increase_ in merc from a mining step at coords
                damage = self.get_damage(coords)
                self.world_state[coords[0], coords[1], -2] += damage
                self.agent_merc_history[agent_id] += damage
                self.agent_merc_cumulative[agent_id] += damage
                # 3: scale the mining reward if merc is used
                reward *= self.merc_efficiency_scale
        else:
            reward = 0 

        return reward

    def get_grid_type(self, coords):
        x, y = coords
        if y < self.mining_height:
            return MINING
        elif y < self.height:
            return EMPTY
        else: 
            raise Exception("shouldn't be in undefined grid type space")

    def get_mining_reward(self, coords):
        prob_of_success = self.mining_probs[coords[0], coords[1]]
        return np.random.binomial(n=1.0, p=prob_of_success)

    def get_damage(self, coords):
        prev_merc_amt = self.world_state[coords[0], coords[1], -2]
        damage = self.beta * np.log(prev_merc_amt + 4)
        return damage

    def update_mining_probs(self):
        self.mining_probs *= (self.scarcity_scale ** self.world_state[:, : , -3])
        return None
    
    def get_cumulative_mining_amount(self, agent_id=None):
        if agent_id is None:
            return np.sum(self.world_state[:, :self.mining_height, :self.num_agents])
        else:
            return np.sum(self.world_state[:, :self.mining_height, agent_id])

    def get_cumulative_merc_amount(self, agent_id=None):
        if agent_id is None:
            return np.sum(self.agent_merc_cumulative[:])
        else:
            return self.agent_merc_cumulative[agent_id]
    
    def get_total_mine_actions(self, agent_id=None):
        if agent_id is None:
            return np.sum(self.total_mine_actions[:])
        else:
            return self.total_mine_actions[agent_id]
    
    def get_total_merc_actions(self, agent_id=None):
        if agent_id is None:
            return np.sum(self.total_merc_actions[:])
        else:
            return self.total_mine_actions[agent_id]
    
    def get_observations(self):
        return [np.concatenate([self.agent_positions, self.evictions[:,np.newaxis]], axis=1) for _ in range(self.num_agents)]

    def evict(self):
        # 3: evict most contaminating agent that is on mining side to empty side
        shuffled_agent_indices = np.arange(self.num_agents)
        np.random.shuffle(shuffled_agent_indices)
        # 3: choose to evict the agent with max merc damage 
        # 3: default eviction is still random (first one)
        max_merc_ind = shuffled_agent_indices[0]
        max_merc = self.agent_merc_history[max_merc_ind ]
        max_merc_coords = self.agent_positions[max_merc_ind , :]
        for ind in shuffled_agent_indices:
            val = self.agent_merc_history[ind]
            if val > max_merc:
                max_merc = val
                max_merc_ind = ind
                max_merc_coords = self.agent_positions[max_merc_ind, :]
        if self.get_grid_type(max_merc_coords) == MINING: 
            while True:
                # loop until an open empty spot is found
                x = np.random.choice(self.width)
                y_relative = np.random.choice(self.mining_height)
                y = 9
                # y = self.height - y_relative - 1
                # check that its the empty side
                assert(self.get_grid_type((x, y)) == EMPTY)
                if self.world_state[x, y, -1] < 0:
                    # successful empty side spot found; update positions
                    self.world_state[x, y, -1] = max_merc_ind
                    # set previous position to -1
                    self.world_state[max_merc_coords[0], max_merc_coords[1], -1] = -1
                    self.agent_positions[max_merc_ind, :] = x, y
                    # note the eviction
                    self.evictions = np.zeros(
                        (self.num_agents,), dtype=np.int32)
                    self.evictions[max_merc_ind] = 1
                    return max_merc_ind
        return None

    def reset_mining_probs(self):
        """Resets mining probabilities. Can choose when this is done."""
        self.mining_probs = (self.mining_prob_bounds[1] -
            self.mining_prob_bounds[0]) * np.random.random(
            size=(self.width, self.height)) + self.mining_prob_bounds[0]
        self.mining_probs[:, self.mining_height:] = 0

    def reset(self):
        # reset the state to initial state
        self.step_count = 0
        if self.reset_mining_probs_every_ep:
            self.reset_mining_probs()
        # initialize world state - for each grid theres a mining/farming history
        # for each agent, and indication of which agent if any is in the grid (0
        # to NUM_AGENTS-1 to represent each agent, and -1 if no agent)
        # 3: index -1 that gives the agent_id in location or -1 if no agent
        # 3: index -2 additional slice that gives the contamination
        # 3: index -3 additional slice that gives the mining actions/time step 
            # this faciliates updating for scarcity
        self.world_state = np.zeros(
            (self.width, self.height, self.num_agents + 3), dtype=np.int32)
        # initialize with "no agents" anywhere
        self.world_state[:, :, -1] = -1
        # evictions in previous timestep
        self.evictions = np.zeros((self.num_agents,), dtype=np.int32)
        # history of contamination for each agent
        self.agent_merc_history = np.zeros(
            (self.num_agents,), dtype=np.int32)
        self.agent_merc_cumulative = np.zeros(
            (self.num_agents,), dtype=np.int32)
        self.total_mine_actions= np.zeros(
            (self.num_agents,), dtype=np.int32)
        self.total_merc_actions= np.zeros(
            (self.num_agents,), dtype=np.int32)
        # initialize agent positions
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        for ind in range(self.num_agents):
            while True:
                x = np.random.choice(self.width)
                # 3: CHANGED: initialize as miners
                y = np.random.choice(self.mining_height)
                # check to see another agent hasn't beenn initialized in that position
                if self.world_state[x, y, -1] < 0:
                    # update agent position
                    self.world_state[x,  y, -1] = ind
                    self.agent_positions[ind, :] = x, y
                    break
        #drawnow.figure(figsize=(7,7))
        obs = self.get_observations()
        return obs

    def update_drawnow(self):
        rgb_arr = self.get_rgb()
        plt.imshow(rgb_arr)

    def render(self, filename=None, stop_on_close=True):
        """ Creates an image of the map to plot or save."""
        if filename is None:
            drawnow.drawnow(self.update_drawnow, stop_on_close=stop_on_close)
        else:
            rgb_arr = self.get_rgb()
            plt.imshow(rgb_arr, interpolation='nearest')
            plt.savefig(filename)

    def get_rgb(self):
        """Returns rgb map of space"""
        rgb_arr = np.zeros((self.width, self.height, 3))
        for i in range(self.width):
            for j in range(self.height):
                grid_type = self.get_grid_type((i, j))
                if grid_type == MINING:
                    contam_amount = self.world_state[i, j, -2]
                    scale = (500-contam_amount)/500
                    rgb_val = [x * scale for x in DEFAULT_COLOURS['M']]
                    rgb_val = DEFAULT_COLOURS['M']
                    rgb_val[1] *= scale
                    rgb_arr[i, j, :] = rgb_val
                # 3: removed farming
                else:
                    rgb_arr[i, j, :] = DEFAULT_COLOURS[' ']

        for agent_ind in range(self.num_agents):
            x, y = self.agent_positions[agent_ind]
            rgb_arr[x, y, :] = DEFAULT_COLOURS[str(agent_ind)]

        rgb_arr = np.around(rgb_arr)
        return rgb_arr.astype(int)

