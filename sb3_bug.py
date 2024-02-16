import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env


class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50, 800))
        self.action_space = spaces.Box(low=-1, high=1, shape=(50,))

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


env = CustomEnv()
check_env(env)

for i in range(1000):
    model = A2C("MlpPolicy", env, verbose=1).learn(1000)
    model.save("a2c_trader")
    del model
