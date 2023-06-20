from qutip_qip.compiler import GateCompiler, Instruction

import numpy as np

class RydbergCz3LevelCompiler(GateCompiler):
    """ Compiler for :obj:`.RydbergCz3Level`.

    unit_pulse_strength: MHz
    unit_gate_time: \mu s

    Supported native gate: "CZ".
    """

    def __init__(
        self,
        num_qubits,
        params,
        global_phase = 0.0,
        pulse_dict = None,
    ):
        super().__init__(
            num_qubits,
            params = params,
            pulse_dict = pulse_dict,
        )
        self.gate_compiler.update(
            {
                "CZ": self._cz_compiler
            }
        )
        self.global_phase = global_phase

    def _cz_compiler(
        self, 
        gate, 
        #op_label, 
        #param_label, 
        args
    ):
        q0 = gate.controls[0]
        q1 = gate.targets[0]
        coeff_del_p = self.params["delta_p_amp"]
        coeff_omg_p = self.params["omega_p_amp"]
        coeff_omg_r = self.params["omega_r_amp"]
        coeff_brr = self.params["B"]
        coeff_tgate = self.params["gate_time"]
        
        #tlist = np.linspace(0.0, coeff_tgate, 100)
        tlist = 1.0
        pulse_info = [
                ("del_p" + str(q0), coeff_del_p),
                ("del_p" + str(q1), coeff_del_p),
                ("omg_p" + str(q0), coeff_omg_p),
                ("omg_p" + str(q1), coeff_omg_p),
                ("omg_r" + str(q0), coeff_omg_r),
                ("omg_r" + str(q1), coeff_omg_r),
                ("brr", coeff_brr)
                ]

        return [Instruction(gate, tlist, pulse_info)]

    #def cz_compiler(self, gate, args):
        



