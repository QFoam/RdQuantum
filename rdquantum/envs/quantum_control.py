import gymnasium as gym
from gymnasium import spaces

from qutip import basis, sigmax, sigmaz, qeye
from qutip.measurement import measure

import numpy as np

class QuantumControlEnv(gym.Env):
    
    def __init__(
        self, 
        QuantumCircuit, 
        QuantumProcessor, 
        render_mode=None
    ):
        metadata = {"render_modes": ["human"], "render_fps": 4}
        self.quantumprocessor = QuantumProcessor(quantumcircuit=QuantumCircuit())
        self.observation_space = self.quantumprocessor.observation_space
        self.action_space = self.quantumprocessor.action_space

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

    def step(
        self, 
        action, 
    ):
        old_observation, reward, terminated = self.quantumprocessor.run_rl_step(action = action)
        info = self._get_info()

        return old_observation, reward, terminated, False, info
