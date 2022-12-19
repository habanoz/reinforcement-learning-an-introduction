#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# Modified by 2023 huseyinabanox@gmail.com
#######################################################################

import matplotlib
import numpy as np

from chapter08.maze_world import MazeWorld

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


# a wrapper class for parameters of dyna algorithms
class DynaParams:
    def __init__(self, max_steps=3000, switch_time=1000, max_episodes=200):
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

        self.max_steps = max_steps

        self.switch_time = switch_time

        self.max_episodes = max_episodes


class RegularDynaModel:
    def __init__(self):
        self.known_transitions = dict()

    def feed(self, s, a, sp, r):
        self.known_transitions[(s, a)] = (r, sp)

    def sample(self):
        # random previously observed state, action pair
        s, a = random.choice(list(self.known_transitions.keys()))

        # use model to determine next reward and state
        r, sp = self.known_transitions[(s, a)]

        return s, a, sp, r

    def bonus(self, s, a):
        return 0


class TimeDynaModel(RegularDynaModel):
    def __init__(self, kappa=1e-4):
        super().__init__()
        self.kappa = kappa
        self.time = 0
        self.model = dict()

    def feed(self, s, a, sp, r):
        self.time += 1

        if tuple(s) not in self.model.keys():
            self.model[tuple(s)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in list(range(4)):  # action space
                if action_ != a:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(s)][action_] = [list(s), 0, 1]

        self.model[tuple(s)][a] = [list(sp), r, self.time]

    def sample(self):
        state_index = random.choice(range(len(self.model.keys())))
        s = list(self.model)[state_index]

        action_index = random.choice(range(len(self.model[s].keys())))
        a = list(self.model[s])[action_index]
        sp, r, time = self.model[s][a]

        r += self.bonus(s,a)

        return s, a, sp, r

    def bonus(self, s, a):
        if s not in self.model:
            return 0
        if a not in self.model[s]:
            return 0

        sp, r, time = self.model[s][a]
        return self.kappa * np.sqrt(self.time - time)


class AlternateTimeDynaModel(TimeDynaModel):
    def __init__(self, kappa=1e-4):
        super().__init__(kappa)

    def sample(self):
        state_index = random.choice(range(len(self.model.keys())))
        s = list(self.model)[state_index]

        action_index = random.choice(range(len(self.model[s].keys())))
        a = list(self.model[s])[action_index]

        sp, r, time = self.model[s][a]

        return s, a, sp, r


def dyna_q(q, model, env, dyna_params):
    sp = env.reset()[0]['agent']
    sp = tuple(sp)

    for t in range(dyna_params.max_steps):
        # a) s = current (non-terminal) state
        s = sp

        # b) A epsilon-greedy(S, Q)
        if np.random.binomial(1, dyna_params.epsilon) == 1:
            a = np.random.choice(env.action_space.n)
        else:
            values = [q[s[0], s[1], a] + model.bonus(s, a) for a in range(env.action_space.n)]
            a = np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

        # c) take action a and observe reward r, next state sp
        obs, r, done, _, _ = env.step(a)

        sp = obs['agent']
        sp = tuple(sp)

        # reward_sum += r
        # if len(reward_hist) <= t:
        #     reward_hist.append(reward_sum)
        # else:
        #     # incremental average
        #     reward_hist[t] = reward_hist[t] + (reward_sum - reward_hist[t]) / run

        # d) Q(S,A) = Q(S,A) + alpha [R + gamma * maxa Q(sp,a) - Q(S,A)]
        q[s[0], s[1], a] += dyna_params.alpha * (r + dyna_params.gamma * np.max(q[sp[0], sp[1], :]) - q[s[0], s[1], a])

        # e) Model(s,a) = R,sp assuming deterministic environment
        model.feed(s, a, sp, r)

        # f) loop repeat n times
        for j in range(dyna_params.planning_steps):
            # sample transitions from model
            sr, ar, spr, rr = model.sample()
            q[sr[0], sr[1], ar] += dyna_params.alpha * (
                    rr + dyna_params.gamma * np.max(q[spr[0], spr[1], :]) - q[sr[0], sr[1], ar])

        if done:
            return t

    return dyna_params.max_steps


# wrapper function for changing maze
# @maze: a maze instance
# @dynaParams: several parameters for dyna algorithms
def changing_maze(maze, dyna_params, model):
    # track the cumulative rewards
    rewards = np.zeros((dyna_params.runs, dyna_params.max_steps))
    cumulative_episode_lengths = np.zeros((dyna_params.runs, dyna_params.max_episodes))

    for run in tqdm(range(dyna_params.runs)):
        q_values = np.zeros(maze.q_size)

        # set old obstacles for the maze
        maze.switchBlocks1()
        switched2 = False

        steps = 0
        last_steps = steps
        episodes = 0
        cumulative_episode_length = 0

        while steps < dyna_params.max_steps:
            # play for an episode
            steps += dyna_q(q_values, model, maze, dyna_params)

            cumulative_episode_length += steps

            # update cumulative rewards
            rewards[run, last_steps: steps] = rewards[run, last_steps]
            rewards[run, min(steps, dyna_params.max_steps - 1)] = rewards[run, last_steps] + 1
            cumulative_episode_lengths[run, episodes] = cumulative_episode_length

            last_steps = steps

            episodes += 1

            if not switched2 and steps > dyna_params.switch_time:
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
    blocking_maze = MazeWorld(initial_agent_position=[3, 5], columns=9, rows=6,
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
    rewards, episode_lengths = changing_maze(blocking_maze, dyna_params, RegularDynaModel())
    rewards2, episode_lengths2 = changing_maze(blocking_maze, dyna_params, TimeDynaModel())

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)

    plt.plot(rewards, label="DynaQ")
    plt.plot(rewards2, label="DynaQ+")
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths, label="DynaQ")
    plt.plot(episode_lengths2, label="DynaQ+")
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.legend()

    plt.savefig('../images/example_8_4.png')
    plt.close()


# Figure 8.5, ShortcutMaze
def figure_8_5():
    # set up a shortcut maze instance
    blocking_maze = MazeWorld(initial_agent_position=[3, 5], columns=9, rows=6,
                              blocks=[[i, 3] for i in range(1, 9)], blocks2=[[i, 3] for i in range(1, 8)],
                              render_mode=None
                              )

    # set up parameters
    dyna_params = DynaParams(max_steps=6000, switch_time=3000, max_episodes=500)
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 50
    dyna_params.runs = 5

    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-3

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

    plt.savefig('../images/figure_8_5.png')
    plt.close()


# Figure 8.4, BlockingMaze
def example_8_4():
    # set up a blocking maze instance
    blocking_maze = MazeWorld(initial_agent_position=[3, 5], columns=9, rows=6,
                              blocks=[[i, 3] for i in range(0, 8)], blocks2=[[i, 3] for i in range(1, 9)],
                              render_mode=None
                              )

    # set up parameters
    dyna_params = DynaParams(max_episodes=500)
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 10
    dyna_params.runs = 20

    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-4

    # play
    # play
    rewards, episode_lengths = changing_maze(blocking_maze, dyna_params, RegularDynaModel())
    rewards2, episode_lengths2 = changing_maze(blocking_maze, dyna_params, TimeDynaModel())
    rewards3, episode_lengths3 = changing_maze(blocking_maze, dyna_params, AlternateTimeDynaModel())

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)

    plt.plot(rewards, label="DynaQ")
    plt.plot(rewards2, label="DynaQ+")
    plt.plot(rewards3, label="DynaQ+A")
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths, label="DynaQ")
    plt.plot(episode_lengths2, label="DynaQ+")
    plt.plot(episode_lengths3, label="DynaQ+A")
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.legend()

    plt.savefig('../images/example_8_4.png')
    plt.close()


# Figure 8.4, BlockingMaze
def example_8_4_2():
    # set up a blocking maze instance
    blocking_maze = MazeWorld(initial_agent_position=[3, 5], columns=9, rows=6,
                              blocks=[[i, 3] for i in range(0, 8)], blocks2=[[i, 3] for i in range(1, 9)],
                              render_mode=None
                              )

    dyna_params = DynaParams(max_steps=6000, switch_time=3000, max_episodes=500)
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 50
    dyna_params.runs = 5

    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-3

    # play
    rewards, episode_lengths = changing_maze(blocking_maze, dyna_params, RegularDynaModel())
    rewards2, episode_lengths2 = changing_maze(blocking_maze, dyna_params, TimeDynaModel())
    rewards3, episode_lengths3 = changing_maze(blocking_maze, dyna_params, AlternateTimeDynaModel())

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)

    plt.plot(rewards, label="DynaQ")
    plt.plot(rewards2, label="DynaQ+")
    plt.plot(rewards3, label="DynaQ+A")
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths, label="DynaQ")
    plt.plot(episode_lengths2, label="DynaQ+")
    plt.plot(episode_lengths3, label="DynaQ+A")
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.legend()

    plt.savefig('../images/example_8_4_2.png')
    plt.close()

if __name__ == '__main__':
    # figure_8_2()
    # figure_8_4()
    # figure_8_5()
    # example_8_4()
    example_8_4_2()
