from typing import Optional, Tuple, Callable

import gymnasium as gym 
import qutip as qt
from qutip.qip.circuit import QubitCircuit, Gate
from qutip import tensor, basis, bell_state, ket2dm, expect, fidelity
from qutip.measurement import measure, measure_observable

from rdquantum.quantumcircuit import QuantumCircuit

import numpy as np

class BellStatePrep(QuantumCircuit):
    def __init__(
        self, 
    ):
        """ Quantum circuit for Bell states
        """
        self.num_qubits = 2
        self.get_control_circuit()
        #self.init_state = '01' 
        #self.target_state = bell01

        """ RL parameters
            - observation: initial state (constant)
            - action: depends on the quantum system
        """
        self.observation_space = gym.spaces.Discrete(1, start=1) 

        super().__init__()

    def get_control_circuit(
        self, 
    ):
        """ Control circuit
                  gp1  gp2  gp3 
            |C> --|H|-- o -------
                        |
            |T> --|H|--|Z|--|H|--
        """
        self.cc = {
                "group1": QubitCircuit(N=self.num_qubits),
                "group2": QubitCircuit(N=self.num_qubits),
                "group3": QubitCircuit(N=self.num_qubits)
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
        return self

    def run_cc(
        self, 
        init_state: str,
        ideal_hadamard: bool = True,
        ideal_control_z: bool = True
    ) -> qt.Qobj:
        self.init_state = tensor(
            basis(2, int(init_state[0])), basis(2, int(init_state[1]))
        )
        self.target_state = self._map_to_Bell_state(init_state)

        if ideal_hadamard and ideal_control_z:
            gs1 = self.cc["group1"].run(state=self.init_state)
            gs2 = self.cc["group2"].run(state=gs1)
            self.final_state = self.cc["group3"].run(state=gs2)
        else:
            self.final_state = tempRydbergCz3Level(init_state)

        return self.final_state

    def run_rc(
        self,
        init_state: qt.Qobj,
    ):
        #self.rc = QubitCircuit(N=self.num_qubits, num_cbits=self.num_qubits)
        """ To do
            Need to consider the case self.backend = quantumdevice
        """
        self.target_state = self._map_to_Bell_state("01")
        prob = qt.expect(qt.ket2dm(self.target_state), init_state)
        measurement_outcome = np.random.choice(2, 1, p=[(1-prob), prob])[0]
        reward = self._get_reward(measurement_outcome)
        return measurement_outcome, reward

    def _map_to_Bell_state(self, init_state):
        bell_map = {
            '00': '00',
            '01': '10',
            '10': '01',
            '11': '11'
        }
        return bell_state(bell_map[init_state])

    def _get_reward(
        self,
        measurement_outcome
    ) -> list:
        """ Bell state projection measurement
        """
        m = measurement_outcome
        reward = [-1 + 2*m]
        return reward

    def _get_fidelity(self) -> float:
        """ Bell fidelity
        """
        self.Bell_fidelity = fidelity(self.final_state, self.target_state)
        return self.Bell_fidelity
