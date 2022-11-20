import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype

from chapter11.bairds_env import BairdsEnv

features = np.array([[2, 0, 0, 0, 0, 0, 0, 1],
                     [0, 2, 0, 0, 0, 0, 0, 1],
                     [0, 0, 2, 0, 0, 0, 0, 1],
                     [0, 0, 0, 2, 0, 0, 0, 1],
                     [0, 0, 0, 0, 2, 0, 0, 1],
                     [0, 0, 0, 0, 0, 2, 0, 1],
                     [0, 0, 0, 0, 0, 0, 1, 2]
                     ])


# equation (10.3)
def q_hat(s, a, w):
    return np.dot(w, x(s, a))


def x(s, a):
    return features[s, :]


def q_learn_tabular(env, alpha=0.1, gamma=0.9, epsilon=0.1):
    s, _ = env.reset()
    q_hist = []
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    done = False
    while not done:
        if np.random.binomial(1, epsilon) == 1:
            a = np.random.randint(env.action_space.n)
        else:
            max_q_val = np.max(Q[s:])
            a = np.random.choice(np.where(Q[s:] == max_q_val)[0])

        sp, r, done, _, _ = env.step(a)

        Q[s, a] += alpha * (r + gamma * np.max(Q[sp:]) - Q[s, a])
        q_hist.append(np.array(Q))

        s = sp

    return Q


def q_learn_semi_gradient(env, alpha=0.01, gamma=0.99):
    s, _ = env.reset()
    w_hist = []
    W = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=dtype('float64'))

    done = False
    while not done:
        # binary action selection using binomial distribution with success rate of 1/7
        # this does not work, if a new action is added to action space...
        a = np.random.binomial(1, 1 / 7)

        sp, r, done, _, _ = env.step(a)

        max_qp_val = max([q_hat(sp, 0, W), q_hat(sp, 1, W)])

        derivative_of_q = x(s, a)

        W += alpha * (r + gamma * max_qp_val - q_hat(s, a, W)) * derivative_of_q

        w_hist.append(np.array(W))

        s = sp

    return w_hist


def epsilon_greedy_action_selection(W, env, epsilon, s):
    if np.random.binomial(1, epsilon) == 1:
        a = env.random_action()
    else:
        q_vals = np.array([q_hat(s, a, W) for a in range(env.action_space.n)])
        max_q_val = max(q_vals)
        a = np.random.choice(np.where(q_vals == max_q_val)[0])
    return a


def run():
    env = BairdsEnv(episode_length=1000)
    w_hist = np.array(q_learn_semi_gradient(env))

    for i in range(len(features) + 1):
        plt.plot(w_hist[:, i], label='w' + str(i + 1))
    plt.xlabel('Steps')
    plt.ylabel('Weight value')
    plt.title('semi-gradient Q-Learning')
    plt.legend()
    # plt.show()

    plt.savefig('../images/exercise_11_3.png')
    plt.close()


if __name__ == '__main__':
    run()
