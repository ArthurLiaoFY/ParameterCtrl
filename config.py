# RL
q_learning_kwargs = {
    "action_mapping_dict": {
        idx: action
        for idx, action in enumerate(
            [
                (act1, act2)
                for act1 in [0, -20, -10, -2, 2, 10, 20]
                for act2 in [0, -500, -100, -10, 10, 100, 500]
            ]
        )
    },
    # -------------------------------
    "learning_rate": 0.1,
    "explore_rate": 0.5,
    "learning_rate_min": 0.03,
    "explore_rate_min": 0.03,
    "learning_rate_decay": 0.999,
    "explore_rate_decay": 0.999,
    "discount_factor": 0.99,
    "fully_explore_step": 2000,
}

ddpg_kwargs = {
    # -------------------------------
    "state_dim": 6,
    "action_dim": 2,
    # -------------------------------
    "batch_size": 256,
    "learning_rate": 1e-2,
    "learning_rate_min": 3e-4,
    "learning_rate_decay_factor": 0.999,
    "discount_factor": 0.99,
    "jitter_noise": 0.05,
    "jitter_noise_min": 1e-5,
    "jitter_noise_decay_factor": 1 - 1e-4,
    "tau": 0.01,
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
    "step_per_episode": 201,
    "env_kwargs": env_kwargs,
    "q_learning_kwargs": q_learning_kwargs,
    "ddpg_kwargs": ddpg_kwargs,
}
