ddpg_kwargs = {
    # -------------------------------
    "state_dim": 6,
    "action_dim": 2,
    # -------------------------------
    "batch_size": 256,
    "learning_rate": 1e-2,
    "learning_rate_min": 3e-4,
    "learning_rate_decay_factor": 1 - 1e-3,
    "discount_factor": 0.99,
    "jitter_noise": 0.2,
    "jitter_noise_min": 1e-5,
    "jitter_noise_decay_factor": 1 - 2e-4,
    "tau": 0.001,
}

ddqn_kwargs = {
    # -------------------------------
    "state_dim": 6,
    "action_dim": 2,
    # -------------------------------
    "batch_size": 256,
    "learning_rate": 1e-2,
    "learning_rate_min": 3e-4,
    "learning_rate_decay_factor": 1 - 1e-3,
    "discount_factor": 0.99,
    "jitter_noise": 0.2,
    "jitter_noise_min": 1e-5,
    "jitter_noise_decay_factor": 1 - 2e-4,
    "tau": 0.001,
}

# Replay buffer
replay_buffer_kwargs = {
    # -------------------------------
    "buffer_size": 1e7,
    "replay_buffer_dir": "./buffer_data",
}

# Env
env_kwargs = {
    # ---------------
    "init_Ca": 1.04,
    "init_Cb": 0.8,
    "init_Tr": 140.52,
    "init_Tk": 139.10,
    "init_F": 21.01,
    "init_Q": -1234.44,
    # ---------------
    "ideal_Ca": 0.70,
    "ideal_Cb": 0.60,
    "ideal_Tr": 127.25,
    "ideal_Tk": 124.39,
    # ---------------
    "noise": 0.1,
    # ---------------
    "upper_F": 100.0,
    "lower_F": 5.0,
    "upper_Q": 0.0,
    "lower_Q": -8500.0,
}

# Train

training_kwargs = {
    "n_episodes": 1000,
    "step_per_episode": 51,
    "early_stop_patience": 10,
    "step_loss_tolerance": 1e-2,
    "env_kwargs": env_kwargs,
    "replay_buffer_kwargs": replay_buffer_kwargs,
    "ddpg_kwargs": ddpg_kwargs,
}
