from gymnasium.envs.registration import register

register(
     id="rdquantum/QubitStatePrep-v2023.04.11",
     entry_point="rdquantum.envs:QubitStatePrepEnv",
     max_episode_steps=2,
)

register(
     id="rdquantum/HamiltonianTrainer-v2023.04.22",
     entry_point="rdquantum.envs:HamiltonianTrainerEnv",
     max_episode_steps=2,
)