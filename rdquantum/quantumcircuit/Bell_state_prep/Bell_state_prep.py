from gymnasium import spaces
import qutip as qt

from rdquantum.quantumcircuit import QuantumCircuit

class Rydberg_Cz_3Level(QuantumCircuit):
    def __init__(self, range_amp, range_gate_time):
        observation_space = spaces.Discrete(1, start=1) 
        self.action_space = spaces.Dict(
            {
                'omega_p_amp': spaces.Box(-1, 1, shape=(), dtype=np.float32)
                'omega_r_amp': spaces.Box(value_eps, 1, shape=(), dtype=np.float32),
                'delta_p_amp': spaces.Box(value_eps, 1, shape=(), dtype=np.float32),
                'gate_time': spaces.Box(value_eps, 1, shape=(), dtype=np.float32)
            }
        )
        super().__init__(observation_space, action_space)

    def _get_control_circuit(self, control_params: dict):
        return cc
    
    def _get_reward_circuit(self):
        return rc
    
    def _get_initail_state(self):
        return initial_state

    def _action_to_control_params(self, action) -> dict:
        return control_params

    def _measurement_outcome_to_reward(self, measurement_outcome) -> list:
        return reward

    def _get_fidelity(self, final_state) -> float:
        return fidelity
