import qutip  as qt

class Rydberg_Cz_3Level():
    def __init__(
            self
            ):
        """ RL parameters
        """
        self.action_space = spaces.Dict(
            {
                'omega_p_amp': spaces.Box(-1, 1, shape=(), dtype=np.float32)
                'omega_r_amp': spaces.Box(-1, 1, shape=(), dtype=np.float32),
                'delta_p_amp': spaces.Box(-1, 1, shape=(), dtype=np.float32),
                'gate_time': spaces.Box(-1, 1, shape=(), dtype=np.float32)
            }
        )

    def gate(
            self,
            ):
        return results
