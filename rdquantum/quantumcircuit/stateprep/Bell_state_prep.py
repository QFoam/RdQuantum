from typing import Optional, Tuple, Callable

from gymnasium import spaces
import qutip as qt
from qutip.qip.circuit import QubitCircuit, Gate
from qutip import tensor, basis, bell_state, ket2dm, expect
from qutip.measurement import measure, measure_observable

from rdquantum.quantumcircuit import QuantumCircuit
#from rdquantum.simulator.rydberg_atom import Rydberg_Cz_3Level

import numpy as np

class BellStatePrep(QuantumCircuit):
    def __init__(
            self, 
            backend: str = "simulator", #quantumdevice/simulator
            processor: str = "RydbergCz3Level",
            range_amp: tuple[float, str] = [100, "MHz"], 
            range_gate_time: tuple[float, str] = [1, "mus"]
            ):
        self.backend = backend
        self.processor = processor
        
        """ Quantum circuit for Bell states
        """
        self.num_qubit = 2
        self.init_state = '01' 
        #self.target_state = bell01

        """ RL parameters
            - observation: initial state (constant)
            - action: depends on the quantum system
        """
        self.observation_space = spaces.Discrete(1, start=1) 
        #self.action_space = self.system.action_space
        super().__init__()

    def quantum_circuit(
            self, 
            ):
        """ Control circuit
                  gp1  gp2  gp3 
            |C> --|H|-- o -------
                        |
            |T> --|H|--|Z|--|H|--
        """
        self.cc = {
                "group1": QubitCircuit(N=self.num_qubit),
                "group2": QubitCircuit(N=self.num_qubit),
                "group3": QubitCircuit(N=self.num_qubit)
                }
        self.cc["group1"].add_gate("SNOT", targets=0)
        self.cc["group1"].add_gate("SNOT", targets=1)
        self.cc["group2"].add_gate("CZ", controls=0, targets=1) 
        self.cc["group3"].add_gate("SNOT", targets=1)

        """ Reward circuit
            |C> --|    |
                  |Bell|
            |T> --|    |
        """
        self.rc = QubitCircuit(N=self.num_qubit, num_cbits=self.num_qubit)
        return self

    def run_cc(
            self, 
            init_state: str,
            ideal_hadamard: bool = True,
            ideal_control_z: bool = True
            ):
        self.init_state = tensor(basis(2, int(init_state[0])), basis(2, int(init_state[1])))
        self.target_state = self._map_to_Bell_state(init_state)

        #group1
        if ideal_hadamard:
            gs1 = self.cc["group1"].run(state=self.init_state)
        else:
            raise Exception("Sorry, currently we only support ideal Hadamard gate")

        #group2
        if ideal_control_z:
            gs2 = self.cc["group2"].run(state=gs1)
        else:
            self.processor.load_circuit(self.cc["group2"])
            gs2 = self.processor.run_state(gs1)

        #group3
        if ideal_hadamard:
            self.final_state = self.cc["group3"].run(state=gs2)
        else:
            raise Exception("Sorry, currently we only support ideal Hadamard gate")

        return self.final_state

    def run_rc(self):
        """ To do
            Need to consider the case self.backend = quantumdevice
        """
        prob = expect(ket2dm(self.target_state), self.final_state)
        self.measurement_outcome = np.random.choice(2, 1, p=[(1-prob), prob])[0]
        return self.measurement_outcome

    def _action_to_control_params(self, action) -> dict:
        return control_params

    def _map_to_Bell_state(self, init_state):
        bell_map = {
                '00': '00',
                '01': '10',
                '10': '01',
                '11': '11'
                }
        return bell_state(bell_map[init_state])

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
