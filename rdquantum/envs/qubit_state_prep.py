import gymnasium as gym
from gymnasium import spaces

from qutip import basis, sigmax, sigmaz, qeye
from qutip.measurement import measure

import numpy as np

class QubitStatePrepEnv(gym.Env):
    
    def __init__(self, render_mode=None):
        metadata = {"render_modes": ["human"], "render_fps": 4}

        # Observation is the measurement outcome of \sigma_Z {1, -1}. with {0, 1} maps to {1, -1}
        self.observation_space = spaces.Discrete(2)

        # # The following dictionary maps abstract observations from `self.observation_space` to the measurement outcome.
        # self._obs_to_measurement = {
        #     0: np.int64(1)
        #     1: np.int64(-1)
        # }

        # Action is sampled by the Gaussian policy distribution (\mu, \sigma^2) within the range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_info(self):
        return self._measurement_outcome

    def _get_obs(self):
        return self._m_sigmaZ    

    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)

        self._measurement_outcome = None
        self._m_sigmaZ = None

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        a = action

        # The unitary matrix U(\theta)
        U = np.cos(np.pi * a) * qeye(2) - 1j * np.sin(np.pi * a) * sigmax()

        # Initial state
        Rho_init = basis(2,0)

        # Perform gate operation U
        Rho_final = U * Rho_init

        # Measurement outcome of \sigma_Z after the gate operation U
        self._measurement_outcome = measure(Rho_final, sigmaz())
        self._m_sigmaZ = self._measurement_outcome[0]

        terminated = False
        observation = self._get_obs()
        reward = -observation
        info = self._get_info()

        return observation, reward, terminated, False, info