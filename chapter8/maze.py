#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib

from chapter8.grid_world_row_first import GridWorldEnvRowFirst

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy


# a wrapper class for parameters of dyna algorithms
class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0


# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.action_space.n)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward


# Time-based model for planning in Dyna-Q+
class TimeModel:
    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, maze, time_weight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.maze = maze

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in list(range(self.maze.action_space.n)):
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(state)][action_] = [list(state), 0, 1]

        self.model[tuple(state)][action] = [list(next_state), reward, self.time]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward


# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_q(q_value, model, maze, dyna_params):
    state = tuple(maze.reset()[0]['agent'])
    steps = 0
    done = False
    while not done:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        obs, reward, done, _, _ = maze.step(action)
        next_state = tuple(obs['agent'])

        # Q-Learning update
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state

        # check whether it has exceeded the step limit
        if steps > 3000:
            break

    return steps


# play for an episode for prioritized sweeping algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
# @return: # of backups during this episode
def prioritized_sweeping(q_value, model, maze, dyna_params):
    state = maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while state not in maze.GOAL_STATES:
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # get the priority for current state action pair
        priority = np.abs(reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                          q_value[state[0], state[1], action])

        if priority > dyna_params.theta:
            model.insert(priority, state, action)

        # start planning
        planning_step = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_step < dyna_params.planning_steps and not model.empty():
            # get a sample with highest priority from the model
            priority, state_, action_, next_state_, reward_ = model.sample()

            # update the state action value for the sample
            delta = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - \
                    q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta

            # deal with all the predecessors of the sample state
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + dyna_params.gamma * np.max(q_value[state_[0], state_[1], :]) -
                                  q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_step += 1

        state = next_state

        # update the # of backups
        backups += planning_step + 1

    return backups


# wrapper function for changing maze
# @maze: a maze instance
# @dynaParams: several parameters for dyna algorithms
def changing_maze(maze, dyna_params):
    # set up max steps
    max_steps = 3000

    # track the cumulative rewards
    rewards = np.zeros((dyna_params.runs, 2, max_steps))
    cumulative_episode_lengths = np.zeros((dyna_params.runs, 2, 200))

    for run in tqdm(range(dyna_params.runs)):
        # set up models
        models = [TrivialModel(), TimeModel(maze, time_weight=dyna_params.time_weight)]

        # initialize state action values
        q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]

        for i in range(len(dyna_params.methods)):
            # print('run:', run, dyna_params.methods[i])

            # set old obstacles for the maze
            maze.switchBlocks1()
            switched2 = False

            steps = 0
            episodes = 0
            cumulative_episode_length = 0
            last_steps = steps
            while steps < max_steps:
                # play for an episode
                steps += dyna_q(q_values[i], models[i], maze, dyna_params)

                cumulative_episode_length += steps

                # update cumulative rewards
                rewards[run, i, last_steps: steps] = rewards[run, i, last_steps]
                rewards[run, i, min(steps, max_steps - 1)] = rewards[run, i, last_steps] + 1
                cumulative_episode_lengths[run, i, episodes] = cumulative_episode_length

                last_steps = steps

                episodes += 1

                if not switched2 and steps > 1000:
                    switched2 = True
                    # change the obstacles
                    maze.switchBlocks2()

    # averaging over runs
    rewards = rewards.mean(axis=0)
    cumulative_episode_lengths = cumulative_episode_lengths.mean(axis=0)

    return rewards, cumulative_episode_lengths


# Figure 8.4, BlockingMaze
def figure_8_4():
    # set up a blocking maze instance

    blocking_maze = GridWorldEnvRowFirst(initial_agent_position=[3, 5], columns=9, rows=6,
                       blocks=[[i, 3] for i in range(0, 8)], blocks2=[[i, 3] for i in range(1, 9)],
                       render_mode=None
                       )

    # set up parameters
    dyna_params = DynaParams()
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 10
    dyna_params.runs = 20

    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-4

    # play
    rewards, episode_lengths = changing_maze(blocking_maze, dyna_params)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)

    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(len(dyna_params.methods)):
        plt.plot(episode_lengths[i, :], label=dyna_params.methods[i])
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.legend()

    plt.savefig('../images/figure_8_4.png')
    plt.close()


# Check whether state-action values are already optimal
def check_path(q_values, maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    max_steps = 14 * maze.resolution * 1.2
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        state, _ = maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False
    return True


if __name__ == '__main__':
    # figure_8_2()
    figure_8_4()
    # figure_8_5()
    # example_8_4()
