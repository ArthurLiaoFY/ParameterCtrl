agent_kwargs = {
    # -------------------------------
    "batch_size": 128,
    "learning_rate": 1e-2,
    "learning_rate_min": 3e-4,
    "learning_rate_decay_factor": 1 - 1e-3,
    "discount_factor": 0.99,
    "jitter_noise": 0.1,
    "jitter_noise_min": 1e-7,
    "jitter_noise_decay_factor": 1 - 2e-5,
    "tau": 0.001,
    # -------------------------------
}

# Replay buffer
replay_buffer_kwargs = {
    # -------------------------------
    "buffer_size": 1e6,
    "replay_buffer_dir": "./buffer_data",
    "prioritized_sampler": {
        "alpha": 1.0,
        "beta": 1.0,
    },
}

cartpole_env_kwargs = {
    # -------------------------------
    "state_dim": 4,
    "action_dim": 2,
    # -------------------------------
}

# Env
cstr_env_kwargs = {
    # -------------------------------
    "state_dim": 6,
    "action_dim": 2,
    # -------------------------------
    "init_Ca": 1.04,
    "init_Cb": 0.8,
    "init_Tr": 140.52,
    "init_Tk": 139.10,
    "init_F": 21.01,
    "init_Q": -1234.44,
    # -------------------------------
    "ideal_Ca": 0.70,
    "ideal_Cb": 0.60,
    "ideal_Tr": 127.25,
    "ideal_Tk": 124.39,
    # -------------------------------
    "noise": 0.01,
    # -------------------------------
    "upper_F": 100.0,
    "lower_F": 5.0,
    "upper_Q": 0.0,
    "lower_Q": -8500.0,
    # -------------------------------
}

# Train

training_kwargs = {
    "n_episodes": 10000,
    "step_per_episode": 21,
    "early_stop_patience": 10,
    "step_loss_tolerance": 1e-2,
    "inference_each_k_episode": 2500,
    "cartpole_env_kwargs": cartpole_env_kwargs,
    "cstr_env_kwargs": cstr_env_kwargs,
    "replay_buffer_kwargs": replay_buffer_kwargs,
    "agent_kwargs": agent_kwargs,
}
