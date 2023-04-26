import gymnasium as gym
from gymnasium import spaces

from qutip import basis, sigmax, sigmaz, qeye
from qutip.measurement import measure

from rdquantum.hamiltonian.pulsegen import GaussianPulse

import numpy as np

class HamiltonianTrainerEnv(gym.Env):
    
    def __init__(self, Hamiltonian, render_mode=None):
        metadata = {"render_modes": ["human"], "render_fps": 4}
        self.Hamiltonian = Hamiltonian()
        self.observation_space = self.Hamiltonian.observation_space
        self.action_space = self.Hamiltonian.action_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_info(self):
        return {'tset': None}

    def _get_obs(self):
        # Convert measurement outcome to observations
        observation = self._measurement_outcome
        return observation

    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)

        self._measurement_outcome = self.observation_space.sample()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Get measurement outcome
        prob, self._measurement_outcome, reward = self.Hamiltonian.measurement(action)
        terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info