import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from chapter8.grid_world_row_first import GridWorldEnvRowFirst


class DynaParams:
    def __init__(self, runs=50, max_steps=3000, planning_steps=10):
        self.runs = runs
        self.max_steps = max_steps
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


class TimeDynaModel():
    def __init__(self, kappa=1e-4):
        super().__init__()
        self.kappa = kappa
        self.time = 0
        self.model = dict()

    def feed(self, s, a, r, sp):
        self.time += 1

        if s not in self.model.keys():
            self.model[s] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in list(range(4)):  # action space
                if action_ != a:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[s][action_] = [s, 0, 1]

        self.model[s][a] = [sp, r, self.time]

    def sample(self):
        state_index = random.choice(range(len(self.model.keys())))
        s = list(self.model)[state_index]

        action_index = random.choice(range(len(self.model[s].keys())))
        a = list(self.model[s])[action_index]
        sp, r, time = self.model[s][a]

        r += self.kappa * np.sqrt(self.time - time)

        return s, a, r, sp


def start():
    env = GridWorldEnvRowFirst(initial_agent_position=[3, 5], columns=9, rows=6,
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

    # plt.show()
    plt.savefig('../images/figure_8_4_grid_world_dyna_method.png')

    # plt.savefig('./images/figure_2_4.png')
    plt.close()


def run_dyna(env, dyna_params, model, initial_seed=None):
    q = defaultdict(float)

    reward_hist = np.zeros((dyna_params.runs, dyna_params.max_steps))
    episode_lengths = np.zeros((dyna_params.runs, 200))

    for run in tqdm.tqdm(range(dyna_params.runs)):
        steps = 0
        last_steps = 0
        episode_count = 0

        env.switchBlocks1()
        switched2 = False

        while steps < dyna_params.max_steps:
            steps += dyna_q(dyna_params, env, model, q)

            episode_lengths[run, episode_count] = steps
            episode_count += 1

            reward_hist[run, last_steps: steps] = reward_hist[run, last_steps]
            reward_hist[run, min(steps, dyna_params.max_steps - 1)] = reward_hist[run, last_steps] + 1

            last_steps = steps

            if not switched2 and steps > 1000:
                env.switchBlocks2()
                switched2 = True

    ##

    reward_hist = reward_hist.mean(axis=0)
    episode_lengths = episode_lengths.mean(axis=0)

    return reward_hist, episode_lengths


def dyna_q(dyna_params, env, model, q):
    sp = env.reset()[0]['agent']
    sp = tuple(sp)


    for t in range(dyna_params.max_steps):
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

        # reward_sum += r
        # if len(reward_hist) <= t:
        #     reward_hist.append(reward_sum)
        # else:
        #     # incremental average
        #     reward_hist[t] = reward_hist[t] + (reward_sum - reward_hist[t]) / run

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
            return t

    return dyna_params.max_steps


if __name__ == '__main__':
    start()
