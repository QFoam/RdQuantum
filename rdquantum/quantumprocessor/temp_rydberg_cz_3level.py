import qutip as qt
from gymnasium import spaces

from rdquantum.simulator.pulsegen import GaussianPulse

import numpy as np

class TempRydbergCz3Level():
    def __init__(
        self, 
        quantumcircuit,
        solver=qt.mesolve, 
        r_amp=500, 
        r_gate_time=10
    ):
        self.quantumcircuit = quantumcircuit
        
        # Physical system
        self.num_energy_level = 5

        ## Computing basis
        ket00 = qt.tensor(qt.basis(self.num_energy_level,0), qt.basis(self.num_energy_level,0))
        ket01 = qt.tensor(qt.basis(self.num_energy_level,0), qt.basis(self.num_energy_level,1))
        ket10 = qt.tensor(qt.basis(self.num_energy_level,1), qt.basis(self.num_energy_level,0))
        ket11 = qt.tensor(qt.basis(self.num_energy_level,1), qt.basis(self.num_energy_level,1))

        ## Hadamard gate
        Had = np.zeros((self.num_energy_level,self.num_energy_level))
        Had[0][0] = 1
        Had[0][1] = 1
        Had[1][0] = 1
        Had[1][1] = -1
        Had = qt.Qobj(Had/np.sqrt(2))

        ## Target Bell state, rho_bell10 = 1/sqrt(2) * (|01> + |10>)
        I = qt.qeye(self.num_energy_level)
        rho0101 = qt.tensor(I, Had) * qt.ket2dm(ket01) * qt.tensor(I, Had)
        rho1010 = qt.tensor(I, Had) * qt.ket2dm(ket10) * qt.tensor(I, Had)
        rho0110 = qt.tensor(I, Had) * (ket10 * ket01.dag()) * qt.tensor(I, Had)
        # self.dm_bell10 = qt.tensor(I, Had) * qt.ket2dm(1/np.sqrt(2) * (ket01 + ket10)) * qt.tensor(I, Had)
        self.dm_bell10 = qt.tensor(I, Had) * qt.ket2dm(1/np.sqrt(2) * (ket01 + ket10)) * qt.tensor(I, Had)

        self.rho_bell = [rho0101, rho1010, rho0110]
        self.rho_init = qt.tensor(Had, Had) * qt.ket2dm(ket01) * qt.tensor(Had, Had)        # Initial state |01>
        self.gamma_p = 1/0.155          # Decay rate of state |p> (1/\mu s)
        self.gamma_r = 1/540            # Decay rate of state |r> (1/\mu s)

        # Method
        self.solver = solver

        # RL parameters
        self.r_amp = r_amp              # MHz
        self.r_gate_time = r_gate_time  # \mu s

        self.observation_space = self.quantumcircuit.observation_space
        value_eps = 1e-5 # for mumerical stability
        self.action_space = spaces.Dict(
            {
            'omega_p_amp': spaces.Box(value_eps, 1, shape=(), dtype=np.float32),
            'omega_r_amp': spaces.Box(value_eps, 1, shape=(), dtype=np.float32),
            'delta_p_amp': spaces.Box(value_eps, 1, shape=(), dtype=np.float32),
            'gate_time': spaces.Box(value_eps, 1, shape=(), dtype=np.float32)
            }
        )

    def _action_to_control_params(self, action):
        control_params = {
            'omega_p': GaussianPulse(
                amp = self.r_amp * action['omega_p_amp'], 
                gate_time = self.r_gate_time * action['gate_time'], 
                tau = 0.165 * self.r_gate_time * action['gate_time']
                ),
            'omega_r': self.r_amp * action['omega_r_amp'],
            'delta_p': self.r_amp * action['delta_p_amp'],
            'gate_time': self.r_gate_time * action['gate_time'],
            'times': np.linspace(0.0, self.r_gate_time * action['gate_time'], 100)
        }
        return control_params

    def _get_hamiltonian(self, control_params):
        # Pulses
        omega_p = control_params['omega_p']
        omega_r = control_params['omega_r']
        delta_p = control_params['delta_p']

        I = qt.qeye(self.num_energy_level)
        H_omega_p = (np.pi) * ( 
            qt.basis(self.num_energy_level,2)*qt.basis(self.num_energy_level,1).dag() 
            + qt.basis(self.num_energy_level,1)*qt.basis(self.num_energy_level,2).dag() 
            )
        H_omega_r = (np.pi) * omega_r * ( 
            qt.basis(self.num_energy_level,3)*qt.basis(self.num_energy_level,2).dag() 
            + qt.basis(self.num_energy_level,2)*qt.basis(self.num_energy_level,3).dag() 
            )
        H_delta_p = (2*np.pi) * delta_p * ( 
            qt.basis(self.num_energy_level,2)*qt.basis(self.num_energy_level,2).dag() 
            )
  
        # (MHz) Strength of Rydberg states interaction
        B = (2*np.pi * 500)
        Brr = np.sqrt(B) * ( qt.basis(self.num_energy_level,3)*qt.basis(self.num_energy_level,3).dag() )
        Brr = qt.tensor(qt.Qobj(Brr), qt.Qobj(Brr))

        Hamiltonian = [
            [qt.tensor(H_omega_p, I) + qt.tensor(I, H_omega_p), omega_p], 
            [qt.tensor(H_omega_r, I) + qt.tensor(I, H_omega_r), '1'], 
            [qt.tensor(H_delta_p, I) + qt.tensor(I, H_delta_p), '1'], 
            [Brr, '1']
            ]

        return Hamiltonian

    def _get_decay_terms(self, gamma_p=1/0.155, gamma_r=1/540):
        # gamma_p: (1/mu s) population decay rate of the Rydberg state
        # gamma_r: (1/mu s) population decay rate of the P state
        c_ops = []
        I = qt.qeye(self.num_energy_level)
        
        # |p>
        L0p = np.sqrt(1/16 * gamma_p) * ( qt.basis(self.num_energy_level,0)*qt.basis(self.num_energy_level,2).dag() )
        c_ops.append(qt.tensor(qt.Qobj(L0p), I))
        c_ops.append(qt.tensor(I, qt.Qobj(L0p)))

        L1p = np.sqrt(1/16 * gamma_p) * ( qt.basis(self.num_energy_level,1)*qt.basis(self.num_energy_level,2).dag() )
        c_ops.append(qt.tensor(qt.Qobj(L1p), I))
        c_ops.append(qt.tensor(I, qt.Qobj(L1p)))
        
        Ldp = np.sqrt(7/8 * gamma_p) * ( qt.basis(self.num_energy_level,4)*qt.basis(self.num_energy_level,2).dag() )
        c_ops.append(qt.tensor(qt.Qobj(Ldp), I))
        c_ops.append(qt.tensor(I, qt.Qobj(Ldp)))
        
        # |r>
        L0r = np.sqrt(1/32 * gamma_r) * ( qt.basis(self.num_energy_level,0)*qt.basis(self.num_energy_level,3).dag() )
        c_ops.append(qt.tensor(qt.Qobj(L0r), I))
        c_ops.append(qt.tensor(I, qt.Qobj(L0r)))

        L1r = np.sqrt(1/32 * gamma_r) * ( qt.basis(self.num_energy_level,1)*qt.basis(self.num_energy_level,3).dag() )
        c_ops.append(qt.tensor(qt.Qobj(L1r), I))
        c_ops.append(qt.tensor(I, qt.Qobj(L1r)))

        Lpr = np.sqrt(1/2 * gamma_r) * ( qt.basis(self.num_energy_level,2)*qt.basis(self.num_energy_level,3).dag() )
        c_ops.append(qt.tensor(qt.Qobj(Lpr), I))
        c_ops.append(qt.tensor(I, qt.Qobj(Lpr)))
        
        Ldr = np.sqrt(7/16 * gamma_r) * ( qt.basis(self.num_energy_level,4)*qt.basis(self.num_energy_level,3).dag() )
        c_ops.append(qt.tensor(qt.Qobj(Ldr), I))
        c_ops.append(qt.tensor(I, qt.Qobj(Ldr)))

        return c_ops

    def run_rl_step(
        self, 
        action, 
        e_ops=[]
    ):
        prob, measurement_outcome, reward = self.measurement(action)
        observation = 1
        terminated = True
        
        return observation, reward, terminated

    def _solve(self, action, e_ops=[]):
        self.control_params = self._action_to_control_params(action)
        self.hamiltonian = self._get_hamiltonian(self.control_params)
        self.decay_terms = self._get_decay_terms(self.gamma_p, self.gamma_r)

        # Solve
        times = self.control_params['times']
        results = self.solver(
            H = self.hamiltonian,
            rho0 = self.rho_init,
            tlist = times,
            c_ops = self.decay_terms,
            e_ops = e_ops,
            options = qt.Options(nsteps=100000, rhs_reuse=False)
            )
        return results

    def measurement(self, action):
        results = self._solve(action, e_ops=self.dm_bell10)

        prob = results.expect[0][-1]
        measurement_outcome = np.random.choice(2, 1, p=[(1-prob), prob])[0]
        reward = -1 + 2*measurement_outcome

        return prob, measurement_outcome, reward
        
    def evaluation(self, action):
        results = self._solve(action, e_ops=self.rho_bell)

        bell1 = results.expect[0][-1]
        bell2 = results.expect[1][-1]
        bell3 = results.expect[2][-1]
        fidelity = 1/2 * (bell1 + bell2) + np.absolute(bell3)
        
        return fidelity
