import random
import time
from collections import defaultdict
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from chapter8.grid_world import GridWorldEnv


class DynaParams:
    def __init__(self, runs=20, planning_steps=10):
        self.runs = runs
        self.planning_steps = planning_steps
        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 1.0
        self.kappa = 1e-4


class RegularDynaModel:
    def __init__(self):
        self.known_transitions = dict()

    def feed(self, s, a, r, sp):
        self.known_transitions[(s, a)] = (r, sp)

    def sample(self):
        # random previously observed state, action pair
        s, a = random.choice(list(self.known_transitions.keys()))

        # use model to determine next reward and state
        r, sp = self.known_transitions[(s, a)]

        return s, a, r, sp


class TimeDynaModel(RegularDynaModel):
    def __init__(self, kappa=1e-4):
        super().__init__()
        self.kappa = kappa
        self.tau = dict()
        self.time = -1

    def feed(self, s, a, r, sp):
        self.time += 1
        super().feed(s, a, r, sp)
        self.tau[(s, a)] = self.time

    def sample(self):
        s, a, r, sp = super().sample()
        return s, a, r + self.kappa * np.sqrt(self.time - self.tau[(s, a)]), sp


def start():
    env = GridWorldEnv(initial_agent_position=[3, 5], width=9, height=6,
                       blocks=[[i, 3] for i in range(0, 8)], blocks2=[[i, 3] for i in range(1, 9)],
                       render_mode=None
                       # render_mode="human"
                       )

    env.render()

    reward_hist, episode_lengths = run_dyna(env, dyna_params=DynaParams(), model=RegularDynaModel())
    reward_hist2, episode_lengths2 = run_dyna(env, dyna_params=DynaParams(), model=TimeDynaModel())

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(reward_hist, label='DynaQ')
    plt.plot(reward_hist2, label='DynaQ+')

    plt.xlabel('Steps')
    plt.ylabel('Cumulative reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths, label='DynaQ')
    plt.plot(episode_lengths2, label='DynaQ+')

    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.legend()

    plt.show()

    # plt.savefig('./images/figure_2_4.png')
    plt.close()


def run_dyna(env, dyna_params, model, initial_seed=None):
    q = defaultdict(float)

    reward_hist = []
    episode_length = 0
    episode_lengths = []

    sp = env.reset(seed=initial_seed)[0]['agent']
    sp = tuple(sp)

    for run in tqdm.tqdm(range(1, dyna_params.runs + 1)):
        reward_sum = 0
        env.switchBlocks1()

        for t in range(3000):
            # a) s = current (non-terminal) state
            s = sp

            # b) A epsilon-greedy(S, Q)
            if np.random.random_sample() > dyna_params.epsilon:
                q_values = list(map(lambda x: q[(s, x)], list(range(env.action_space.n))))
                q_values_array = np.array(q_values)
                a = np.random.choice(np.where(q_values_array == q_values_array.max())[0])
            else:
                a = np.random.randint(0, env.action_space.n)

            # c) take action a and observe reward r, next state sp
            obs, r, done, _, _ = env.step(a)

            sp = obs['agent']
            sp = tuple(sp)

            reward_sum += r
            if len(reward_hist) <= t:
                reward_hist.append(reward_sum)
            else:
                # incremental average
                reward_hist[t] = reward_hist[t] + (reward_sum - reward_hist[t]) / run

            # d) Q(S,A) = Q(S,A) + alpha [R + gamma * maxa Q(sp,a) - Q(S,A)]
            next_q_values = map(lambda x: q[(sp, x)], list(range(env.action_space.n)))
            q[(s, a)] = q[(s, a)] + dyna_params.alpha * (r + dyna_params.gamma * max(next_q_values) - q[(s, a)])

            # e) Model(s,a) = R,sp assuming deterministic environment
            model.feed(s, a, r, sp)

            # f) loop repeat n times
            for j in range(dyna_params.planning_steps):
                # sample transitions from model
                sr, ar, rr, spr = model.sample()
                next_q_values = map(lambda x: q[(spr, x)], list(range(env.action_space.n)))
                # update q values
                q[(sr, ar)] = q[(sr, ar)] + dyna_params.alpha * (
                        rr + dyna_params.gamma * max(next_q_values) - q[(sr, ar)])

            if done:
                sp = env.reset()[0]['agent']
                sp = tuple(sp)
                episode_lengths.append(episode_length)
                episode_length = 0
            else:
                episode_length += 1

            if t == 1000:
                env.switchBlocks2()

        episode_lengths.append(episode_length)

    return reward_hist, episode_lengths


if __name__ == '__main__':
    start()
