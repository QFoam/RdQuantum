import qutip  as qt
import qutip.qip.device import Model, ModelProcessor

from copy import deepcopy

class RydbergCz3Level(ModelProcessor):
    def __init__(
            self,
            num_qubit=2, 
            dims=5
            **params
            ):
        model = RydbergCz3LevelModel(
                num_qubits=num_qubits,
                dims = dims,
                correct_global_phase = True
                **params
                )
        super(RydbergCz3Level, self).__init__(
                model = model,
                correct_global_phase = correct_global_phase
                )
        self.correct_global_phase = correct_global_phase
        

        """ RL parameters
        self.action_space = spaces.Dict(
            {
                'omega_p_amp': spaces.Box(-1, 1, shape=(), dtype=np.float32)
                'omega_r_amp': spaces.Box(-1, 1, shape=(), dtype=np.float32),
                'delta_p_amp': spaces.Box(-1, 1, shape=(), dtype=np.float32),
                'gate_time': spaces.Box(-1, 1, shape=(), dtype=np.float32)
            }
        )
        """

        """ Hamiltonians
        """
        self.controls = {}
        self.controls.update(
                )

        def load_circuit(
                self, 
                quantumcircuit, 
                schedule_mode="ASAP", 
                compiler=None
                ):
            if compiler is None:
                compiler = RydbergCz3LevelCompiler(
                        self.num_qubits, self.params, global_phase = 0.0
                        )
            tlist, coeff = super().load_circuit(
                    qc, compiler = compiler
                    )
            self.global_phase = compiler.global_phase
            return tlist, coeff

class RydbergCz3LevelModel(Model):
    """ To do: documentation
    """
    def __init__(
            self, 
            num_qubits=2, 
            num_levels = 5,
            **params
            ):
        self.num_qubits = num_qubits
        self.num_levels = num_levels
        self.dims = num_levels
        self.params = {
                "omega_p_amp": 100,
                "omega_r_amp": 175,
                "delta_p_amp": 300,
                "gate_time": 1,
                "B": 500
                }
        self.params.update(deepcopy(params))
        self._controls = self._set_up_controls()

    def _set_up_controls(self):
        """
        Generate the Hamiltonian for the Cz gate using Rydberg atoms with 3 energy levels and save them in the attribute `ctrls`.
        {0, 1, p, r, d}
        """
        controls = {}
        num_qubits = self.num_qubits
        num_levels = self.num_levels
        

        # Delta_p
        for m in range(num_qubits):
            controls["del_p" + str(m)] = (
                    2 * np.pi 
                    * (
                        qt.basis(num_levels,2) * qt.basis(num_levels,2).dag()
                        ),
                    m
                    )  

        # Omega_p
        for m in range(num_qubits):
            controls["omg_p" + str(m)] = (
                    np.pi 
                    * (
                        qt.basis(num_levels,2) * qt.basis(num_levels,1).dag() 
                        + qt.basis(num_levels,1) * qt.basis(num_levels,2).dag()
                        ),
                    m
                    )

        # Omega_r
        for m in range(num_qubits):
            controls["omg_r" + str(m)] = (
                    np.pi
                    * (
                        qt.basis(num_levels,3) * qt.basis(num_levels,2).dag()
                        + qt.basis(num_levels,2) * qt.basis(num_levels,3).dag()
                        ),
                    m
                    )
        # B_rr
        Brr = qt.basis(num_levels,3) * qt.basis(num_levels,3).dag()
        for m in range(num_qubits-1):
            controls["brr"] = (
                    qt.tensor(Brr, Brr),
                    [m, m-1]
                    )

        return controls

    """ To do:
        get_control_latex
    """
