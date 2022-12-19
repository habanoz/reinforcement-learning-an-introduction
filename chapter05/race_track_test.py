import unittest
import numpy as np
from chapter05.race_track import Env, CELL_START_LINE
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

TEST_ENV = 'env2'


class MyTestCase(unittest.TestCase):
    def test_something(self):
        pi = None
        with open(TEST_ENV + '_policy.obj', 'rb') as f:
            pi = pickle.load(f)

        grid = np.loadtxt(TEST_ENV + '.txt')

        for col in range(grid.shape[1]):
            if grid[-1, col] != CELL_START_LINE:
                continue

            if col in []: # skip desired columns
                continue

            self.do_test_play(grid, pi, (grid.shape[0] - 1, col))


    def do_test_play(self, grid, pi, pos0):
        env = Env(grid, pos0)
        s0 = env.reset()

        V = np.copy(grid)

        V[s0[0][0], s0[0][1]] = 4
        (i, j), (a, b) = s0

        a0 = pi[i, j, a, b]
        rp, sp = env.act(a0)

        while sp is not None:
            pos = sp[0]
            V[pos[0], pos[1]] = 4

            (i, j), (a, b) = sp
            ap = pi[i, j, a, b]

            # ap = pi[sp]
            rp, sp = env.act(ap)

        plot_value_function(V, file_name=TEST_ENV + '_demo_e_5_12_' + str(pos0[1]) + '.png')


def plot_value_function(V, file_name='e_5_12_demo.png'):
    plt.figure(figsize=(V.shape[1] * 0.25, V.shape[0] * 0.25))
    fig = sns.heatmap(V, cmap="YlGnBu", cbar=False, linewidths=0.1, linecolor='gray')
    fig.set_title('Path', fontsize=10)
    plt.savefig('../images/' + file_name)


if __name__ == '__main__':
    unittest.main()
