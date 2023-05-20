from gymnasium.envs.registration import register

register(
     id="rdquantum/QubitStatePrep-v2023.04.11",
     entry_point="rdquantum.envs:QubitStatePrepEnv",
     max_episode_steps=2,
)

register(
     id="rdquantum/QuantumControlEnv-v2023.05.19",
     entry_point="rdquantum.envs:QuantumControlEnv",
     max_episode_steps=2,
)
