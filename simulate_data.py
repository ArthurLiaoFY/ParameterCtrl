import copy

from scipy.integrate import odeint

from cstr_model import cstr, np


def simulate_CSTR(
    tc_trace: list,
    simulate_time_step: int = 201,
    simulate_size: int = 1000,
    noise: float = 0.1,
    init_Ca: float = 0.87725294608097,
    init_T: float = 324.475443431599,
    **kwargs
):
    """
    u_traj: Trajectory of input values
    data_simulation: Dictionary of simulation data
    repetitions: Number of simulations to perform
    """

    # creating lists
    Ca_data = []
    T_data = []

    # multiple repetitions
    for _ in range(simulate_size):
        Ca_sim_traj = np.array([init_Ca])
        T_sim_traj = np.array([init_T])

        # main process simulation loop
        for i in range(simulate_time_step - 1):
            # integrate system
            y = odeint(
                func=cstr,
                y0=(Ca_sim_traj[-1].item(), T_sim_traj[-1].item()),
                t=[i, i + 1],
                args=(tc_trace[i],),
            )

            Ca_sim_traj = np.append(
                Ca_sim_traj,
                y[-1][0] + noise * np.random.uniform(low=-1, high=1, size=1) * 0.1,
            )
            T_sim_traj = np.append(
                T_sim_traj,
                y[-1][1] + noise * np.random.uniform(low=-1, high=1, size=1) * 5,
            )

        # data collection
        Ca_data.append(Ca_sim_traj)
        T_data.append(T_sim_traj)

    return np.array(Ca_data), np.array(T_data)
