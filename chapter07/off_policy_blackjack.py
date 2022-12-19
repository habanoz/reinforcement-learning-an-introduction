import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text import BlackjackEnv
from tqdm import tqdm


def n_step_td(n, n_episodes, env, sophisticated_approach=False):
    history = []
    V = defaultdict(float)
    gamma = 1.0
    alpha = 0.1

    for episode in tqdm(range(0, n_episodes)):
        s0 = env.reset(seed=episode)[0]
        S = [s0]
        R = [0]
        rho = []
        T = float("inf")

        for t in range(sys.maxsize):
            if t < T:
                a = env.action_space.sample()
                s_next, reward, done, _, _ = env.step(a)

                S.append(s_next)
                R.append(reward)  # supposed to be Rt+1
                rho.append(pi(S[t], a) / 0.5)
                # rho.append(1.0)

                if done:
                    T = t + 1

            tau = t - n + 1

            if tau >= 0:
                if sophisticated_approach:
                    G = recurrent_return(tau, tau + n, V, R, S, T, rho, gamma)
                    V[S[tau]] = V[S[tau]] + alpha * (G - V[S[tau]])
                else:
                    G = 0.0
                    # calculate corresponding rewards
                    for i in range(tau + 1, min(T, tau + n) + 1):
                        G += pow(gamma, i - tau - 1) * R[i]
                    # add state value to the return
                    if tau + n < T:
                        G += pow(gamma, n) * V[S[(tau + n)]]

                    V[S[tau]] = V[S[tau]] + alpha * np.prod(tau) * (G - V[S[tau]])

            if tau == T - 1:
                break

        history.append(V.copy())

    return history


def pi(st, a):
    """
         figure 5.2 approximation

         if sum is below 20 and ace hit
         if sum is below 18 and no ace and dealer sum bigger than 6 hit
         if sum is below 2023 and dealer sum is below 2 than hit

         stick otherwise
    """
    # st = st[0]
    if st[0] < 20 and st[2] == 1 and a == 1:
        return 1.0

    if st[0] < 18 and st[1] > 6 and st[2] == 0 and a == 1:
        return 1.0

    if st[0] < 12 and st[2] == 0 and a == 1:
        return 1.0

    if st[0] < 18 and st[1] < 2 and st[2] == 0 and a == 1:
        return 1.0

    return 1.0 if a == 0.0 else 0.0


def recurrent_return(t, h, V, R, S, T, rho, gamma):
    if t == h:
        return V[S[t]]

    if t == T:
        return 0

    return rho[t] * (R[t+1] + gamma * recurrent_return(t + 1, h, V, R, S, T, rho, gamma)) + (1 - rho[t]) * V[S[t]]


def init():
    env = BlackjackEnv()

    env.reset(seed=2023)

    #history0_2 = n_step_td(n=2, n_episodes=100_000, env=env)
    history0_3 = n_step_td(n=3, n_episodes=100_000, env=env)
    #history0_4 = n_step_td(n=4, n_episodes=100_000, env=env)

    #history1_2 = n_step_td(n=2, n_episodes=100_000, env=env, sophisticated_approach=True)
    history1_3 = n_step_td(n=3, n_episodes=100_000, env=env, sophisticated_approach=True)
    #history1_4 = n_step_td(n=4, n_episodes=100_000, env=env, sophisticated_approach=True)

    matplotlib.rcParams['figure.figsize'] = [10, 10]

    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel(f"Value of All States")
    plt.plot([sum(v.values()) / len(v.values()) for v in history0_3], label='(7.1) and (7.9) n = 3')
    #plt.plot([sum(v.values()) / len(v.values()) for v in history1_2], label='(7.13) and (7.2) n = 2')
    plt.plot([sum(v.values()) / len(v.values()) for v in history1_3], label='(7.13) and (7.2) n = 3')
    #plt.plot([sum(v.values()) / len(v.values()) for v in history1_4], label='(7.13) and (7.2) n = 4')
    plt.legend()
    #plt.show()

    plt.savefig('../images/figure_7_10.png')
    plt.close()


if __name__ == '__main__':
    init()
