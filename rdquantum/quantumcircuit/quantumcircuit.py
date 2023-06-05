class QuantumCircuit():
    def __init__(self):
        """ Quantum circuit
        """
        self.quantum_circuit()

    def quantum_circuit(self, control_params: dict):
        raise NotImplementedError

    def run_cc(self, init_state):
        raise NotImplementedError

    def run_rc(self, final_state):
        raise NotImplementedError

    def _action_to_control_params(self, action) -> dict:
        raise NotImplementedError

    def _get_reward(self) -> list:
        raise NotImplementedError

    def _get_fidelity(self) -> float:
        raise NotImplementedError

    def _update_quantum_circuit(self, control_params):
        self.quantumcircuit = self._get_quantum_circuit(control_params)

    def run_rl_step(self, action, observation) -> tuple[list, list, bool, dict]:
        # Update self.quantumcircuit
        control_params = _action_to_control_params(action)
        self._update_quantum_circuit(control_params)

        # Run self.quantumcircuit
        self.observation = self._get_obs()
        self.final_state = self.run_cc(self.init_state)
        self.measurement_outcome, self.info = self.run_rc(self.final_state)
        self.reward = self._get_reward()
        self.terminated = True

        return self.observation, self.reward, self.terminated, info

    def evaluate(self, action, observation=None) -> float:
        # Update self.quantumcircuit
        control_params = _action_to_control_params(action)
        self._update_quantum_circuit(control_params)

        # Run self.quantumcircuit
        fidelity = self._get_fidelity()
        
        return fidelity
