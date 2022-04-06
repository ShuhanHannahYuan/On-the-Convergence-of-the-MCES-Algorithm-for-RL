import numpy as np
import copy
from tqdm import tqdm
import datetime
import argparse
import os

ACTION_TO_SIGN = {0: '>', 1: '^', 2: '<', 3: 'v'}

class CliffWalkingEnvTime():
    # time is now part of state, thus the environment becomes a feed-forward MDP
    def __init__(self, width, height, wind_probability, time_limit, wind_only_affect_right_action, time_in_state):
        """
        :param width: width of environment
        :param height: height of environment
        :param wind_probability: the probability that wind happens, when wind happens
        :param wind_only_affect_right_action: if True, then wind only happens when agent moves to the right
        :param time_limit: the time limit of each episode
        :param time_in_state: whether time is part of the state given to the agent
        """
        self.width = width
        self.height = height
        self.goal_state = [self.width - 1, 0]
        self.time_limit = time_limit
        self.wind_probability = wind_probability
        self.wind_only_affect_right_action = wind_only_affect_right_action
        self.time_in_state = time_in_state
    def reset(self, uniform_reset = False):
        # reset the environment, will return the initial state
        if uniform_reset:
            # need to be very careful: when we do uniform reset, we should not init from a cliff state or terminal state
            # for all valid initial states, we should make sure we init in these states
            # with equal probability (this function from last version (incorrectly) included terminal state, and
            # the (0, 0) state is given more probability.
            total_number_init_states = self.width * (self.height-1) + 1
            prob_init_in_task_starting_state = 1/total_number_init_states
            if np.random.uniform() < prob_init_in_task_starting_state:
                x, y = 0, 0
            else:
                x = np.random.randint(self.width)
                y = np.random.randint(1, self.height)
            t = np.random.randint(self.time_limit)
        else:
            x = 0
            y = 0
            t = 0
        if self.time_in_state:
            self.state = [x, y, t] # the starting state. The 3 values are x, y position and current time step
        else:
            self.state = [x, y]
        return copy.deepcopy(self.state)
    def step(self, action):
        # right, up, left, down
        assert action in [0, 1, 2, 3]
        next_state = copy.deepcopy(self.state)
        if action == 0:
            next_state[0] += 1
        elif action == 1:
            next_state[1] += 1
        elif action == 2:
            next_state[0] -= 1
        elif action == 3:
            next_state[1] -= 1
        # now get wind effect
        if self.wind_probability > 0:
            if self.wind_only_affect_right_action: # if wind can only affect "move to right" action
                if action == 0: # effect only possible when action is "move to right"
                    if np.random.uniform() < self.wind_probability:  # decide whether wind affects agent
                        if np.random.uniform() < 0.5:  # decide whether move upwards or downwards
                            next_state[1] += 1
                        else:
                            next_state[1] -= 1
            else: # if wind can affect all actions
                if np.random.uniform() < self.wind_probability:  # decide whether wind affects agent
                    random_val = np.random.uniform()
                    # decide which direction the agent is blown to
                    if random_val < 0.25:
                        next_state[0] += 1
                    elif random_val < 0.5:
                        next_state[1] += 1
                    elif random_val < 0.75:
                        next_state[0] -= 1
                    else:
                        next_state[1] -= 1

        # check special end cases and give rewards
        if self.check_out_of_boundary(copy.deepcopy(next_state)):
            reward = -1
            # move back to within boundary
            if next_state[0] < 0:
                next_state[0] = 0
            if next_state[0] >= self.width:
                next_state[0] = self.width - 1
            if next_state[1] < 0:
                next_state[1] = 0
            if next_state[1] >= self.height:
                next_state[1] = self.height - 1

        # check if reached terminal state, note here it's possible wind got the agent out of boundary
        # the agent bounces back to within boundary, and then arrive in a terminal state
        if self.check_reached_cliff(copy.deepcopy(next_state)):
            reward = -100
            done = True
        elif self.check_reached_goal(copy.deepcopy(next_state)):
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        # separately check time limit
        if self.time_in_state:
            next_state[2] += 1
            if self.check_reached_time_limit(copy.deepcopy(next_state)):
                done = True
        self.state = next_state
        return copy.deepcopy(self.state), reward, done

    def check_out_of_boundary(self, state):
        # check if a state outside boundary
        is_out = (state[0] < 0) or (state[0] >= self.width) or (state[1] < 0) or (state[1] >= self.height)
        return is_out
    def check_reached_cliff(self, state):
        # check if a state is in the cliff area
        in_cliff = (state[1] == 0) and (state[0] != 0) and (state[0] != self.width-1)
        return in_cliff
    def check_reached_goal(self, state):
        reached_goal = (state[0] == (self.width - 1)) and (state[1] == 0)
        return reached_goal
    def check_reached_time_limit(self, state):
        reached_time_limit = (state[2] >= self.time_limit)
        return reached_time_limit

    def choose_action(self,policy_map):
        return policy_map[tuple(self.state)]

    def get_correct_state_and_reward(self, next_state):
        # given a state, check what is the correct next state, and reward (the input might be out of boundary state)
        corrected_state = copy.deepcopy(next_state)

        # check special end cases and give rewards
        if self.check_out_of_boundary(copy.deepcopy(corrected_state)):
            reward = -1
            # move back to within boundary
            if corrected_state[0] < 0:
                corrected_state[0] = 0
            if corrected_state[0] >= self.width:
                corrected_state[0] = self.width - 1
            if corrected_state[1] < 0:
                corrected_state[1] = 0
            if corrected_state[1] >= self.height:
                corrected_state[1] = self.height - 1

        # check if reached terminal state, note here it's possible wind got the agent out of boundary
        # the agent bounces back to within boundary, and then arrive in a terminal state
        if self.check_reached_cliff(copy.deepcopy(corrected_state)):
            reward = -100
            done = True
        elif self.check_reached_goal(copy.deepcopy(corrected_state)):
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        # separately check time limit
        if self.time_in_state:
            corrected_state[2] += 1
            if self.check_reached_time_limit(copy.deepcopy(corrected_state)):
                done = True

        return corrected_state, reward, done

    def get_next_states_and_probabilities(self, state, action):
        # will return a list
        # each entry in the list is a tuple (next_state, reward, done)
        if self.time_in_state: # if time in state, then wind affect all actions
            next_state = copy.deepcopy(state)
            if action == 1:
                next_state[1] += 1
            elif action == 2:
                next_state[0] -= 1
            elif action == 3:
                next_state[1] -= 1
            else:
                next_state[0] += 1

            tuple_list = []
            prob1 = 1 - self.wind_probability
            next_state1 = copy.deepcopy(next_state)
            next_state1, reward1, done1 = self.get_correct_state_and_reward(next_state1)
            tuple_list.append([next_state1, reward1, done1, prob1])

            for i_wind_direction in range(4):
                prob = self.wind_probability/4
                next_state_wind = copy.deepcopy(next_state)
                if i_wind_direction == 0:
                    next_state_wind[0] += 1
                elif i_wind_direction == 1:
                    next_state_wind[1] += 1
                elif i_wind_direction == 2:
                    next_state_wind[0] -= 1
                else:
                    next_state_wind[1] -= 1
                next_state_wind, reward, done = self.get_correct_state_and_reward(next_state_wind)
                tuple_list.append([next_state_wind, reward, done, prob])
            return tuple_list
        else: # if not time in state, and should be wind only affect right action
            next_state = copy.deepcopy(state)
            if action in (1, 2, 3):
                if action == 1:
                    next_state[1] += 1
                elif action == 2:
                    next_state[0] -= 1
                elif action == 3:
                    next_state[1] -= 1
                next_state, reward, done = self.get_correct_state_and_reward(next_state)
                return [[next_state, reward, done, 1],]
            elif action == 0:
                tuple_list = []

                # return 3 tuples
                # state1: wind not affected
                prob1 = 1-self.wind_probability
                next_state1 = copy.deepcopy(state)
                next_state1[0] += 1
                next_state1, reward1, done1 = self.get_correct_state_and_reward(next_state1)
                tuple_list.append([next_state1, reward1, done1, prob1])

                prob2 = self.wind_probability/2
                next_state2 = copy.deepcopy(state)
                next_state2[0] += 1
                next_state2[1] += 1
                next_state2, reward2, done2 = self.get_correct_state_and_reward(next_state2)
                tuple_list.append([next_state2, reward2, done2, prob2])

                prob3 = self.wind_probability/2
                next_state3 = copy.deepcopy(state)
                next_state3[0] += 1
                next_state3[1] -= 1
                next_state3, reward3, done3 = self.get_correct_state_and_reward(next_state3)
                tuple_list.append([next_state3, reward3, done3, prob3])
                return tuple_list

def get_value_with_state_action(env, state, action, value_map):
    # given state and action and value_map, return the computed value for value iteration update
    # the environmnet must have the below function to return a list of transitions
    # each transition is of the form [state, reward, done, prob]
    list_of_transitions = env.get_next_states_and_probabilities(list(state), action)
    cumulative_value = 0

    for transition in list_of_transitions:
        state, reward, done, prob = transition
        if done:
            next_state_value = 0
        else:
            next_state_value = value_map[tuple(state)]
        cumulative_value += prob * (reward + next_state_value)

    return cumulative_value

class EpisodeInfo():
    def __init__(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []
    def store_transition(self, state, action, reward):
        self.state_list.append(tuple(state))
        self.action_list.append(action)
        self.reward_list.append(reward)
    def get_processed_state_action_return(self):
        # when this is called, will compute MC return for every transition tuple stored
        # NOTE: in blackjack, MC return is same for all because reward is only possible when reach terminal state
        # but for cliffwalking it is different

        return_list = []
        current_MC_return = 0
        n_transition = len(self.reward_list)

        for i in range(n_transition - 1, -1, -1):
            current_MC_return += self.reward_list[i]
            return_list.append(current_MC_return)

        return_list.reverse()
        return self.state_list, self.action_list, return_list
def PerformMonteCarloES(world_width, world_height, wind_prob, time_limit,
                        n_episode, wind_only_affect_right_action, time_in_state,
                        multi_update, uniform_start, seed = 0, optimal_q_map=None,
                        visualize=False, log_interval=100, eval_interval=100, n_eval=10):
    action_value_map = { } # this is just Q(s,a)
    policy_map = { } # map playerState to [0, 1, 2, 3] -- right, up, left, down
    returns = { }
    performance = []

    # init env and seed numpy
    env = CliffWalkingEnvTime(world_width, world_height, wind_prob, time_limit,
                              wind_only_affect_right_action, time_in_state)
    np.random.seed(seed)

    state_space = []
    action_space = [0, 1, 2, 3]

    # init state space
    if time_in_state:
        for x in range(world_width):
            for y in range(world_height):
                for time in range(time_limit):
                    player_state = (x, y, time)
                    state_space.append(player_state)
    else:
        for x in range(world_width):
            for y in range(world_height):
                player_state = (x, y)
                state_space.append(player_state)

    # init policy to random actions
    for player_state in state_space:
        policy_action = 0
        for action in action_space:
            random_initial_q = np.random.random()
            action_value_map[(player_state, action)] = random_initial_q

            returns[(player_state, action)] = []
            if random_initial_q > action_value_map[(player_state, policy_action)]:
                policy_action = action
        policy_map[player_state] = policy_action

    """
    we don't set values to -inf, instead, we simply use the random initial q values
    now to deal with infinite loops, 
    """

    avg_abs_diff_list = []
    Q_update_diff_y = []
    Q_update_abs_difference_til_now = []
    for i in tqdm(range(n_episode)):
        # includes running an episode and change policy

        # if i < 3e4:
        #     multi_update_to_use = False
        # else:
        multi_update_to_use = multi_update

        Q_update_abs_difference = GenerateEpisode(env, action_value_map, policy_map, returns, wind_only_affect_right_action,
                        time_in_state, multi_update_to_use, uniform_start, i, time_limit)
        Q_update_abs_difference_til_now += Q_update_abs_difference

        # Evaluate current policy performance by running 1000 episode per 10 episode for the first 1e6 episode
        if (i+1) % eval_interval == 0:
            scores = EvaluatePerformance(env, policy_map, n_eval=n_eval)
            performance.append(np.mean(scores))

        if (i+1) % log_interval == 0:
            if optimal_q_map is not None:
                avg_abs_diff = get_avg_q_abs_diff_to_optimal(state_space, action_space, action_value_map, optimal_q_map, returns=returns)
                avg_abs_diff_list.append(avg_abs_diff)
            else:
                avg_abs_diff_list.append(0)

        if i % log_interval == 0:
            if len(Q_update_abs_difference_til_now) == 0:
                to_add = 0
            else:
                to_add = np.mean(Q_update_abs_difference_til_now)
            Q_update_diff_y.append(to_add)
            Q_update_abs_difference_til_now = []

    if visualize:
        print('current policy: ')
        visualize_policy(env, policy_map, time_in_state)
        print('current q: ')
        visualize_action_value_func(env, action_value_map, time_in_state)
    return avg_abs_diff_list, performance, Q_update_diff_y

def GenerateEpisode(env, action_value_map, policy_map, returns, wind_only_affect_right_action,
                    time_in_state, multi_update, uniform_start, episode, time_limit):
    # init environment and reset environment
    state = env.reset(uniform_start)
    reward = 0
    done = False
    total_reward = 0

    # use this to store the transitions encountered in one episode
    episode_info = EpisodeInfo()

    # when time not in state, it's possible we reach the timelimit and don't have a terminal signal
    # in this case, the data in the episode will not be used.
    if time_in_state:
        use_episode = True
    else:
        use_episode = False

    # start an episode
    for t in range(time_limit):
        # get action, first action is always random
        if t == 0:
            action = np.random.randint(0, 4)
        else:
            action = env.choose_action(policy_map)

        # take a step in the environment
        next_state, reward, done = env.step(action)
        total_reward += reward

        if t == time_limit - 1 and done == False and not time_in_state:
            # this penalty only applies to the case when time is not part of state (this is when loop is possible)
            done = True
            reward = reward - 100

        # multi update: add all s,a pair; first update: only the first one
        episode_info.store_transition(state, action, reward)

        # update state to next state
        state = next_state
        if done:
            use_episode = True
            break

    # update Q table and policy
    if use_episode:
        Q_update_abs_difference = EvaluateAndImprovePolicy(action_value_map, policy_map, returns, episode_info, multi_update, episode)
    else:
        Q_update_abs_difference = []
    return Q_update_abs_difference

def EvaluateAndImprovePolicy(action_value_map, policy_map, returns, episode_info, multi_update, episode):
    Q_update_abs_difference = [] # the abs Q update difference for s-a pairs that are updated for this episode

    state_list, action_list, return_list = episode_info.get_processed_state_action_return()
    if multi_update:
        n_transition_to_use = len(state_list)
    else:
        n_transition_to_use = 1

    already_updated_sa = set()
    for i in range(n_transition_to_use):
        state = state_list[i]
        action = action_list[i]
        MC_return = return_list[i]
        pair = (state, action)
        if pair in already_updated_sa:
            continue
        already_updated_sa.add(pair)

        returns[pair].append(MC_return) # returns is a mapping from s-a pair to list of returns
        n_returns_sa = len(returns[pair]) # currently for this s-a pair, how many valid returns are obtained

        delta_Q = (MC_return - action_value_map[pair]) / (n_returns_sa)
        Q_update_abs_difference.append(abs(delta_Q))

        if n_returns_sa == 1:
            action_value_map[pair] = MC_return
        elif n_returns_sa > 1:
            action_value_map[pair] = action_value_map[pair] + delta_Q
            # action_value_map[pair] =  action_value_map[pair] * 0.99 + MC_return * 0.01
        else:
            quit('n of returns cannot be 0 when computing incremental mean')

        current_policy_action = policy_map[state]

        for act in range(4):
            # update to new action if new action's value is higher
            if action_value_map[(state, act)] > action_value_map[(state, current_policy_action)]:
                current_policy_action = act

        # update policy
        policy_map[state] = current_policy_action
    return Q_update_abs_difference


def EvaluatePerformance(env, policy_map, n_eval=1):
    score_list = []
    for i in range(n_eval):
        # init environment and reset environment
        state = env.reset()
        reward = 0
        done = False
        total_reward = 0

        # start an episode
        for t in range(env.time_limit):
            # get action, first action is always random
            action = env.choose_action(policy_map)

            # take a step in the environment
            next_state, reward, done = env.step(action)
            total_reward += reward

            # update state to next state
            state = next_state
            if done:
                break

        score_list.append(total_reward)
    return score_list

def visualize_policy(env, policy_map, time_in_state, timestep=0):
    for y in range(env.height - 1, -1, -1):
        for x in range(env.width):
            if time_in_state:
                state = (x, y, timestep)
            else:
                state = (x, y)

            if y == 0 and x == env.width-1: # if goal state
                print('T', end='\t')
            elif y==0 and x > 0: # if cliff
                print('X', end='\t')
            else:
                print(ACTION_TO_SIGN[policy_map[state]], end='\t')
        print('\n')

def visualize_value_func(env, value_map, time_in_state, timestep=0):
    for y in range(env.height - 1, -1, -1):
        for x in range(env.width):
            if time_in_state:
                state = (x, y, timestep)
            else:
                state = (x, y)

            if y == 0 and x == env.width-1: # if goal state
                print('T', end='\t')
            elif y==0 and x > 0: # if cliff
                print('  X  ', end='\t')
            else:
                print('%.2f'% value_map[state], end='\t')
        print('\n')

def visualize_action_value_func(env, action_value_map, time_in_state, timestep=0):
    for y in range(env.height - 1, -1, -1):
        for x in range(env.width):
            if time_in_state:
                state = (x, y, timestep)
            else:
                state = (x, y)

            if y == 0 and x == env.width-1: # if goal state
                print('{:^30}'.format('T'), end='\t')
            elif y==0 and x > 0: # if cliff
                print('{:^30}'.format('X'), end='\t')
            else:
                q_val_string = '%.1f/%.1f/%.1f/%.1f' % (action_value_map[(state, 0)],
                      action_value_map[(state, 1)],
                      action_value_map[(state, 2)],
                      action_value_map[(state, 3)])

                print('{:^30}'.format(q_val_string), end='\t')
        print('\n')

def get_avg_q_abs_diff_to_optimal(state_space, action_space, action_value_map, optimal_q_map, q_diff_max=150, returns=None):
    q_abs_diff_to_optimal = []
    for player_state in state_space:
        if player_state[1] == 0 and (player_state[0] > 0):
            continue
        for action in action_space:
            current_q_est = action_value_map[(player_state, action)]
            optimal_q = optimal_q_map[(player_state, action)]
            diff = np.abs(current_q_est - optimal_q)
            if diff > q_diff_max:
                diff = q_diff_max
            q_abs_diff_to_optimal.append(diff)
    return np.mean(q_abs_diff_to_optimal)


def value_iteration(world_width, world_height, wind_prob, time_limit,
        wind_only_affect_right_action, time_in_state, seed = 0, visualize=False):
    action_value_map = { }
    policy_map = { } # map playerState to [0, 1, 2, 3] -- right, up, left, down
    returns = { }
    value_map = {}

    # init env and seed numpy
    env = CliffWalkingEnvTime(world_width, world_height, wind_prob, time_limit,
                              wind_only_affect_right_action, time_in_state)
    np.random.seed(seed)
    state_space = []
    action_space = [0, 1, 2, 3]

    # init state space
    if time_in_state:
        for time in range(time_limit-1, -1, -1):
            for y in range(world_height-1, -1, -1):
                for x in range(world_width-1, -1, -1):
                    player_state = (x, y, time)
                    state_space.append(player_state)
    else:
        for x in range(world_width):
            for y in range(world_height):
                player_state = (x, y)
                state_space.append(player_state)

    # init policy to random actions
    for player_state in state_space:
        policy_action = 0
        for action in action_space:
            random_initial_q = np.random.random()
            action_value_map[(player_state, action)] = random_initial_q
            returns[(player_state, action)] = []
            if random_initial_q > action_value_map[(player_state, policy_action)]:
                policy_action = action
        policy_map[player_state] = policy_action

    # we also should set init action value to minus infinity to prevent loops
    # so we will overwrite the Q value given in the previous step
    for player_state in state_space:
        for action in action_space:
            action_value_map[(player_state, action)] = - np.Infinity

    # init as minus infinity?
    for state in state_space:
        value_map[state] = - np.inf

    for i in range(1000):
        has_change = False
        for state in state_space:
            """
            here we need: 
            1. probability of transition to other states from this state
            2. one step reward 
            3. the current value of another state
            """
            # we now update the value for this state, for each action in this state, we compute a value, and we update to
            # the max value of action
            old_state_value = value_map[state]
            current_max = - np.inf
            for action in action_space:
                action_value = get_value_with_state_action(env, state, action, value_map)
                if action_value > current_max:
                    current_max = action_value
            value_map[state] = current_max
            if old_state_value != current_max:
                has_change = True

        if not has_change:
            print("converged in:", i)
            break

    # set policy
    for state in state_space:
        action_to_choose = 0
        current_max = - np.inf
        for action in range(4):
            action_value = get_value_with_state_action(env, state, action, value_map)
            action_value_map[(state, action)] = action_value # save Q(s,a) value for later use
            if action_value > current_max:
                current_max = action_value
                action_to_choose = action

        policy_map[state] = action_to_choose

    if visualize:
        print('optimal policy: ')
        visualize_policy(env, policy_map, time_in_state)
        print('optimal value: ')
        visualize_value_func(env, value_map, time_in_state)
        print('optimal q: ')
        visualize_action_value_func(env, action_value_map, time_in_state)

    return value_map, action_value_map, policy_map

