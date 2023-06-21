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
        init_state: str = '01',
        ideal_hadamard: bool = True,
        ideal_control_z: bool = True
    ):
        """ Quantum circuit for Bell states
        """
        self.num_qubits = 2
        self.ideal_hadamard = ideal_hadamard
        self.ideal_control_z = ideal_control_z
        self.get_control_circuit()
        self.init_state = init_state 
        self.target_state = self._map_to_Bell_state(self.init_state)

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
        processor,
        init_state: str = None,
    ) -> qt.Qobj:
        if init_state == None:
            init_state = self.init_state
        target_state = self._map_to_Bell_state(init_state)
        init_state = tensor(basis(2, int(init_state[0])), basis(2, int(init_state[1])))
        if self.ideal_hadamard and self.ideal_control_z:
            gs1 = self.cc["group1"].run(state=init_state)
            gs2 = self.cc["group2"].run(state=gs1)
            final_state = self.cc["group3"].run(state=gs2)
        elif self.ideal_hadamard and not self.ideal_control_z:
            gs1 = self.cc["group1"].run(state=init_state)
            gs1_converted = processor.generate_init_processor_state(gs1)
            #processor.pulse_mode = "continuous"
            processor.pulse_mode = "discrete"
            gs2= processor.run_state(
                    init_state = gs1_converted,
                    qc = self.cc["group2"],
                    options=qt.Options(nsteps=100000, rhs_reuse=False)
            ).states[-1]
            gs2_converted = processor.get_final_circuit_state(gs2)
            final_state = self.cc["group3"].run(state=gs2_converted)
        else:
            raise Exception("Sorry, currently we do not support non-ideal Hadamard gate")

        return final_state, target_state

    def run_rc(
        self,
        final_state: qt.Qobj,
        target_state: qt.Qobj
    ):
        """ Reward Circuit
        Currently, we choose Bell measurement as our reward.

        To do:
            - Need to consider the case self.backend = quantumdevice
            - Realistic reward
        """
        prob = qt.expect(qt.ket2dm(target_state), final_state)
        measurement_outcome = np.random.choice(2, 1, p=[(1-prob), prob])[0]
        reward = self._get_reward(measurement_outcome)
        return prob, measurement_outcome, reward

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
