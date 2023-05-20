class QuantumCircuit():
    def __init__(self, observation_space, action_space):
        # RL parameters
        self.observation_space = observation_space
        self.action_space = action_space

        # QuantumCircuit
        self.ControCircuit = self._get_control_circuit()
        self.RewardCircuit = self._get_reward_circuit()
        self.init_state = self._get_initial_state()

    def _get_control_circuit(self, control_params: dict):
        raise NotImplementedError
    
    def _get_reward_circuit(self):
        raise NotImplementedError
    
    def _get_initail_state(self):
        raise NotImplementedError

    def _action_to_control_params(self, action) -> dict:
        raise NotImplementedError

    def _measurement_outcome_to_reward(self, measurement_outcome) -> list:
        raise NotImplementedError

    def _get_fidelity(self, final_state) -> float:
        raise NotImplementedError

    def _update_control_circuit(self, control_params: dict):
        self.ControCircuit = _get_control_circuit(control_params) 

    def run(self, action, observation):
        # Update ControCircuit
        control_params = _action_to_control_params(action)
        self._update_control_circuit(control_params)

        # Run ControCircuit
        final_state =  self.ControCircuit.run(self.init_state)

        # Run RewardCircuit
        measurement_outcome, info = self.RewardCircuit.run(final_state)
        reward = self._measurement_outcome_to_reward(measurement_outcome)

        terminated = True

        return observation, reward, terminated, info

    def evaluate(self, action, observation):
        # Update ControCircuit
        control_params = _action_to_control_params(action)
        self._update_control_circuit(control_params)

        # Run ControCircuit
        final_state =  self.ControCircuit.run(self.init_state)
        fidelity = self._get_fidelity(final_state)
        
        return fidelity
