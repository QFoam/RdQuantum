import gymnasium as gym
from gymnasium import spaces

from qutip import basis, sigmax, sigmaz, qeye
from qutip.measurement import measure

import numpy as np

class QuantumControlEnv(gym.Env):
    
    def __init__(self, QuantumCircuit, render_mode=None):
        metadata = {"render_modes": ["human"], "render_fps": 4}
        self.quantumcircuit = QuantumCircuit
        self.observation_space = self.QuantumCircuit.observation_space
        self.action_space = self.QuantumCircuit.action_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_info(self):
        return {'tset': None}

    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)

        observation = self.observation_space.sample()
        info = self._get_info()

        return observation, info

    def step(self, action, old_observation):
        observation, reward, terminated = self.quantumcircuit.run(action, old_observation)
        info = self._get_info()

        return observation, reward, terminated, False, info
