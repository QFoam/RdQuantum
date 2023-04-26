import qutip as qt

class Hamiltonian():
    def __init__(self, solver=qt.mesolve):
        self.solver = solver

    def _action_to_control_params(self, action):
        return control_params

    def _get_hamiltonian(self):
        return hamiltonian
    
    def _get_decay_terms(self):
        return decay_terms

    def _get_measurement_outcome(self):
        return measurement_outcome

    def _get_reward(self):
        return reward

    def _get_fidelity(self, action):
        return fidelity

    def measurement(self, action, evaluation=False):
        self.control_params = self._action_to_control_params(action)
        self.hamiltonian = self._get_hamiltonian()
        self.decay_terms = self._get_decay_terms()

        results = self.solver(
            H = self.hamiltonian,
            rho0 = ,
            tlist = ,
            c_ops = self.decay_terms,
            options = qt.Options(nsteps=100000, rhs_reuse=False)
        )
        measurement_outcome = self._get_measurement_outcome()
        reward = self._get_reward()

        return reward, measurement_outcome
    
    def evaluation(self, action):
        fidelity = self._get_fidelity(action)
        
        return fidelity