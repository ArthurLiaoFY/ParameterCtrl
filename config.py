# RL
q_learning_kwargs = {
    "action_mapping_dict": {
        idx: action
        for idx, action in enumerate([0, -1, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1])
    },
    # -------------------------------
    "learning_rate": 0.1,
    "explore_rate": 0.5,
    "learning_rate_min": 0.03,
    "explore_rate_min": 0.03,
    "learning_rate_decay": 0.999,
    "explore_rate_decay": 0.999,
    "discount_factor": 0.99,
    "fully_explore_step": 100,
}

# Env
env_kwargs = {
    "init_Ca": 0.87725294608097,
    "init_T": 324.475443431599,
    "init_Tc": 297.0,
    "ideal_Ca": 0.9,
    "ideal_T": 320.0,
    "noise": 0.1,
    "upper_Tc": 300.0,
    "lower_Tc": 290.0,
}

# Train

training_kwargs = {
    "n_episodes": 1000,
    "step_per_episode": 201,
    "env_kwargs": env_kwargs,
    "q_learning_kwargs": q_learning_kwargs,
}
