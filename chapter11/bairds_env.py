####
# huseyinabanox@gmail.com 2022
##

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BairdsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, episode_length=1000):

        self.current_state = None
        self.time_step = None
        self.episode_length = episode_length

        self.window_size_width = 512  # The size of the PyGame window
        self.window_size_height = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Discrete(7)

        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def random_action(self):
        return np.random.randint(2)

    def _get_obs(self):
        return self.current_state

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_state = self.random_select_upper_state()
        self.time_step = -1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def random_select_upper_state(self):
        return np.random.randint(6)

    def dash(self):
        return self.step(0)

    def solid(self):
        return self.step(1)

    def step(self, action):

        if action == 0:  # dashed action
            self.current_state = self.random_select_upper_state()
        else:
            self.current_state = 6

        self.time_step += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, 0, self.time_step >= self.episode_length, False, info

    def render(self):
        print("Not implemented!")

    def _render_frame(self):
        print("Not implemented")

    def close(self):
        pass
