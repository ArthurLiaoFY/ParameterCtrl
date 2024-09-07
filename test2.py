# %%
import matplotlib.pyplot as plt
import numpy as np

from cstr_env import cstr
from plot_fns import plot_simulation
from simulate_data import simulate_CSTR, online_ctrl_Tc

# %%
simulate_size = 100
simulate_time = 201
noise = 0.05
tc_trace = [302] * 100 + [295] * (simulate_time - 100)
Ca_array, T_array = simulate_CSTR(
    tc_trace=tc_trace,
    simulate_size=simulate_size,
    simulate_time_step=simulate_time,
    noise=noise,
)

plot_simulation(Ca_array=Ca_array.T, T_array=T_array.T, tc_trace=tc_trace)

# %%
online_ctrl_Tc()
# %%
