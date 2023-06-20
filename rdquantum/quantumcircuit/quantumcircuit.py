class QuantumCircuit():
    def __init__(self):
        """ Quantum circuit
        """
    def get_control_circuit(self):
        raise NotImplementedError

    def run_cc(self, init_state):
        raise NotImplementedError

    def run_rc(self, init_state):
        raise NotImplementedError

    def _get_reward(self) -> list:
        raise NotImplementedError

    def _get_fidelity(self) -> float:
        raise NotImplementedError
