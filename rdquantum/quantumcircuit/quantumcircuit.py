class QuantumCircuit():
    def __init__(self, observation_space, action_space):
        # RL parameters
        self.observation_space = observation_space
        self.action_space = action_space

        # Pysical system 
        self.init_state = self._get_initial_state()

    def _get_quantum_circuit(self, control_params: dict):
        raise NotImplementedError
    
    def _get_initail_state(self):
        raise NotImplementedError

    def _action_to_control_params(self, action) -> dict:
        raise NotImplementedError

    def _measurement_outcome_to_reward(self, measurement_outcome) -> list:
        raise NotImplementedError

    def _get_fidelity(self) -> float:
        raise NotImplementedError

    def _update_quantum_circuit(self, control_params):
        self.quantumcircuit = self._get_quantum_circuit(control_params)

    def run(self, action, observation) -> tuple(list, list, bool, dict):
        # Update self.quantumcircuit
        control_params = _action_to_control_params(action)
        self._update_quantum_circuit(control_params)

        # Run self.quantumcircuit
        observation, measurement_outcome, info = self.quantumcircuit.run(self.init_state)
        reward = self._measurement_outcome_to_reward(measurement_outcome)

        terminated = True

        return observation, reward, terminated, info

    def evaluate(self, action, observation):
        # Update self.quantumcircuit
        control_params = _action_to_control_params(action)
        self._update_quantum_circuit(control_params)

        # Run self.quantumcircuit
        fidelity = self._get_fidelity()
        
        return fidelity
