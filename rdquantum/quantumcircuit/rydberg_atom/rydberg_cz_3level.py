from rdquantum.quantumcircuit import QuantumCircuit

class Rydberg_Cz_3Level(QuantumCircuit):
    def __init__(self):
        # RL parameters
        observation_space = 
        action_space = 
        super().__init__(observation_space, action_space)

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
