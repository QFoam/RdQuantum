from typing import Optional, Tuple, Callable

from gymnasium import spaces
import qutip as qt
from qutip.qip.circuit import QubitCircuit, Gate
from qutip import tensor, basis

from rdquantum.quantumcircuit import QuantumCircuit
#from rdquantum.simulator.rydberg_atom import Rydberg_Cz_3Level

class BellStatePrep(QuantumCircuit):
    def __init__(
            self, 
            backend: str,
            system: str,
            range_amp: tuple[float, str], 
            range_gate_time: tuple[float, str],
            ):
        self.backend = "simulator"
        
        """ Quantum circuit for Bell states
        """
        self.cc = QubitCircuit(N=2)
        self.rc = QubitCircuit(N=2, num_cbits=2)
        self.init_state = tensor(basis(2,0), basis(2,1))
        #self.target_state = bell01
        self.hadamard = "SNOT"
        self.control_z = "CZ"

        """ RL parameters
            - observation: initial state (constant)
            - action: depends on the quantum system
        """
        self.observation_space = spaces.Discrete(1, start=1) 
        #self.action_space = self.system.action_space
        super().__init__()

    def quantum_circuit(
            self, 
            hadamard: Optional[Callable] = None,
            control_z: Optional[Callable] = None,
            ):
        if hadamard is not None:
            self.hadamard = hadamard
        
        if control_z is not None:
            self.control_z = control_z

        """ Control circuit
            |C> --|H|-- o -------
                        |
            |T> --|H|--|Z|--|H|--
        """
        # Contron circuit
        self.cc.add_gate(self.hadamard, targets=0)
        self.cc.add_gate(self.hadamard, targets=1)
        self.cc.add_gate(self.control_z, controls=0, targets=1) 
        self.cc.add_gate(self.hadamard, targets=1)

        """ Reward circuit
            |C> --|    |
                  |Bell|
            |T> --|    |
        """
        return self

    def _action_to_control_params(self, action) -> dict:
        return control_params

    def _get_reward(self) -> list:
        """ Bell state projection measurement
        """
        m = self.measurement_outcome
        reward = [-1 + 2*m]
        return reward

    def _get_fidelity(self, final_state) -> float:
        """ Bell fidelity
        """
        return fidelity
