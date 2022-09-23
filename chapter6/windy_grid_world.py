#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
import random
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_UP_LEFT = 4
ACTION_UP_RIGHT = 5
ACTION_DOWN_LEFT = 6
ACTION_DOWN_RIGHT = 7
ACTION_NO_MOVE = 8

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]


def step(state, action, use_stochastic_wind):
    i, j = state

    wind = WIND[j] + (random.randint(-1, 1) if use_stochastic_wind and WIND[j] > 0 else 0)

    if action == ACTION_UP:
        return [max(i - 1 - wind, 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - wind, WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - wind, 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - wind, 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_UP_LEFT:
        return [max(i - 1 - wind, 0), max(j - 1, 0)]
    elif action == ACTION_UP_RIGHT:
        return [max(i - 1 - wind, 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN_LEFT:
        return [max(min(i + 1 - wind, WORLD_HEIGHT - 1), 0), max(j - 1, 0)]
    elif action == ACTION_DOWN_RIGHT:
        return [max(min(i + 1 - wind, WORLD_HEIGHT - 1), 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_NO_MOVE:
        return [max(i - wind, 0), j]
    else:
        assert False


# play for an episode
def episode(q_value, actions, use_stochastic_wind):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(actions)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action, use_stochastic_wind)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(actions)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time


def run(actions, file_name, use_stochastic_wind=False):
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(actions)))
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value, actions, use_stochastic_wind))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('../images/{}.png'.format(file_name))
    plt.close()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue

            # if [i, j] == START:
            #   optimal_policy[-1].append('S')
            #   continue

            bestAction = np.argmax(q_value[i, j, :])

            if bestAction == ACTION_UP:
                optimal_policy[-1].append('↑')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('↓')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('←')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('→')
            elif bestAction == ACTION_UP_LEFT:
                optimal_policy[-1].append('⬉')
            elif bestAction == ACTION_UP_RIGHT:
                optimal_policy[-1].append('⬈')
            elif bestAction == ACTION_DOWN_LEFT:
                optimal_policy[-1].append('⬋')
            elif bestAction == ACTION_DOWN_RIGHT:
                optimal_policy[-1].append('⬊')
            elif bestAction == ACTION_NO_MOVE:
                optimal_policy[-1].append('O')

    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))

    plot_path(build_optimum_path_grid(q_value, use_stochastic_wind), file_name)


def build_optimum_path_grid(Q, use_stochastic_wind):
    grid = np.zeros((Q.shape[0], Q.shape[1]))

    i, j = START
    grid[i, j] = 2

    while [i, j] != GOAL:
        best_action = np.argmax(Q[i, j, :])
        next_state = step((i, j), best_action, use_stochastic_wind)
        grid[next_state[0], next_state[1]] = 1
        i, j = next_state[0], next_state[1]
    grid[i, j] = 3
    return grid


def plot_path(grid, file_name):
    fig = sns.heatmap(grid, cmap="YlGnBu", linewidths=0.5, linecolor='black')
    fig.set_title('Path', fontsize=10)
    plt.savefig('../images/{}.png'.format(file_name + "_grid"))
    plt.close()


if __name__ == '__main__':
    run([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT], 'figure_6_3')

    run([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP_LEFT, ACTION_UP_RIGHT, ACTION_DOWN_LEFT,
         ACTION_DOWN_RIGHT], 'figure_6_3_ex_6_9_a')

    run([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP_LEFT, ACTION_UP_RIGHT, ACTION_DOWN_LEFT,
         ACTION_DOWN_RIGHT, ACTION_NO_MOVE], 'figure_6_3_ex_6_9_b')

    run([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT], 'figure_6_3_ex_6_10',
        use_stochastic_wind=True)
