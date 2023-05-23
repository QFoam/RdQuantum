from gymnasium import spaces
import qutip as qt

from rdquantum.quantumcircuit import QuantumCircuit
from rdquantum.simulator.rydberg_atom import Rydberg_Cz_3Level

class BellStatePrep(QuantumCircuit):
    def __init__(
            self, 
            backend: str,
            system: str,
            range_amp: tuple(float, str), 
            range_gate_time: tuple(float, str),
            ):
        self.backend = backend
        self.system = system
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

    def _get_quantum_circuit(self, control_params: dict):
        return quantumcircuit

    def _get_initail_state(self):
        return initial_state

    def _action_to_control_params(self, action) -> dict:
        return control_params

    def _measurement_outcome_to_reward(self, measurement_outcome) -> list:
        return reward

    def _get_fidelity(self, final_state) -> float:
        return fidelity

        # Update ControCircuit
        control_params = _action_to_control_params(action)
        self._update_control_circuit(control_params)

        # Run ControCircuit
        final_state =  self.ControCircuit.run(self.init_state)

        # Run RewardCircuit
        measurement_outcome, info = self.RewardCircuit.run(final_state)
        reward = self._measurement_outcome_to_reward(measurement_outcome)

        terminated = True
