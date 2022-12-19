from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

MAX_SPEED = 4

CELL_EDGE = 0
CELL_START_LINE = 2
CELL_TRACK = 1
CELL_FINISH_LINE = 3

REWARD_FINISH = 0
REWARD_MOVE = -1
REWARD_OUT_OF_TRACK = -100
INITIAL_VALUE = -150


class Env:
    def __init__(self, grid, pos0=None):
        self.grid = grid
        self.pos0 = pos0
        self.position = None
        self.speed = (0, 0)

    def reset(self):
        start_line_indices = np.where(self.grid == CELL_START_LINE)
        selected_index = np.random.randint(0, len(start_line_indices[0]))

        if self.pos0 is None:
            self.position = self.random_start_pos(selected_index, start_line_indices)
        else:
            self.position = self.pos0

        self.speed = (0, 0)

        return self.position, self.speed

    def random_start_pos(self, selected_index, start_line_indices):
        return (start_line_indices[0][selected_index], start_line_indices[1][selected_index])

    def act(self, action):
        # determine new speed
        speedp = (max(min(self.speed[0] + action[0], MAX_SPEED), 0), max(min(self.speed[1] + action[1], MAX_SPEED), 0))

        # if both velocity components are 0, use original value randomly
        if speedp[0] == 0 and speedp[1] == 0:
            speedp = self.speed

        postionp = (self.position[0] - speedp[0], self.position[1] + speedp[1])

        # is it in finish line
        if self.grid[postionp[0], min(postionp[1], self.grid.shape[1] - 1)] == CELL_FINISH_LINE:
            return REWARD_FINISH, None

        # is it out of track
        if postionp[0] < 0 or postionp[1] >= self.grid.shape[1] or self.grid[postionp[0], postionp[1]] == CELL_EDGE:
            s0 = self.reset()
            return REWARD_OUT_OF_TRACK, s0

        self.position = postionp
        self.speed = speedp

        return REWARD_MOVE, (self.position, self.speed)


def action_gen():
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            yield i, j


def play(env, policy):
    states = []
    actions = []
    rewards = []

    s0 = env.reset()
    a0 = policy(s0)

    states.append(s0)
    actions.append(a0)

    rp, sp = env.act(a0)

    while sp is not None:
        rewards.append(rp)
        states.append(sp)

        ap = policy(sp)
        actions.append(ap)

        rp, sp = env.act(ap)

        # print_success_lines(actions, rp, sp, states)

    rewards.append(rp)

    return states, actions, rewards


def print_success_lines(actions, rp, sp, states):
    if sp is None and rp == 0:
        for i, (s, a) in enumerate(zip(states, actions)):
            print("{} -> {}".format(s, a), end=", ")
        print("")


def mc_control(grid, gamma=1.0, max_episode=10_000, env_name='env0'):
    Q = defaultdict(lambda: INITIAL_VALUE)
    C = defaultdict(int)
    pi = np.zeros((grid.shape[0], grid.shape[1], MAX_SPEED + 1, MAX_SPEED + 1), dtype=(int, 2))
    env = Env(grid)

    episode = 0
    for _ in tqdm(range(max_episode)):
        # print("Episode {}".format(episode_count))
        episode += 1

        states, actions, rewards = play(env, policy=lambda s: (np.random.randint(-1, 2), np.random.randint(-1, 2)))
        G = 0
        W = 1

        for st, at, rtp in zip(reversed(states), reversed(actions), reversed(rewards)):
            G = G * gamma + rtp

            stat = (st, at)
            C[stat] = C[stat] + W
            Q[stat] = Q[stat] + (W / C[stat]) * (G - Q[stat])

            action_list = list(action_gen())
            action_values = [Q[(st, a)] for a in action_list]

            act = action_list[np.argmax(action_values)]
            pi[st[0][0], st[0][1], st[1][0], st[1][1]] = act
            if at != act:
                break
            W = W * 9  # w/(1/9), because 9 actions with equal probability

    # np.savetxt('pi.txt', pi)
    with open(env_name + '_policy.obj', 'wb') as f:
        pickle.dump(pi, f)

    V = build_avg_value_function(Q, grid)
    plot_value_function(V, episode, env_name + "_5_3_1_avg.png")

    V = build_max_value_function(Q, grid)
    plot_value_function(V, episode, env_name + "_5_3_1_max.png")


def build_avg_value_function(Q, grid):
    rows, cols = grid.shape
    V = np.zeros(grid.shape)
    for i in range(rows):
        for j in range(cols):
            avg = 0
            n = 0
            for si in range(MAX_SPEED + 1):
                for sj in range(MAX_SPEED + 1):
                    for ai in [-1, 0, 1]:
                        for aj in [-1, 0, 1]:
                            key = (((i, j), (si, sj)), (ai, aj))
                            if key in Q:
                                q = Q[key]
                                n += 1
                                avg = avg + (q - avg) / n
            V[i, j] = avg
    return V


def build_max_value_function(Q, grid):
    rows, cols = grid.shape
    V = np.zeros(grid.shape) - 30
    for i in range(rows):
        for j in range(cols):
            values = [-30]
            for si in range(MAX_SPEED + 1):
                for sj in range(MAX_SPEED + 1):
                    for ai in [-1, 0, 1]:
                        for aj in [-1, 0, 1]:
                            key = (((i, j), (si, sj)), (ai, aj))
                            if key in Q:
                                q = Q[key]
                                values.append(q)
            V[i, j] = max(values)
    return V


def race_track(env_name, gamma=1.0):
    grid = np.loadtxt(env_name + '.txt')
    mc_control(grid, gamma, env_name=env_name, max_episode=50_000)


def plot_value_function(V, episode_count, file_name='5_3_2.png'):
    fig = sns.heatmap(V, cmap="YlGnBu")
    fig.set_title('Value Function ({} iterations)'.format(episode_count), fontsize=10)
    plt.savefig('../images/' + file_name)
    plt.close()


if __name__ == '__main__':
    race_track('env3')
