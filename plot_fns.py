import matplotlib.pyplot as plt
import numpy as np


def plot_simulation(
    Ca_array: np.ndarray,
    T_array: np.ndarray,
    tc_trace: list,
    lower_Tc: float,
    upper_Tc: float,
):
    t = range(len(tc_trace))

    plt.figure(figsize=(8, 5))

    plt.subplot(3, 1, 1)
    plt.plot(t, np.median(Ca_array, axis=1), "r-", lw=3)
    plt.gca().fill_between(
        t, np.min(Ca_array, axis=1), np.max(Ca_array, axis=1), color="r", alpha=0.2
    )
    plt.ylabel("Ca (mol/m^3)")
    plt.xlabel("Time (min)")
    plt.legend(["Concentration of A in CSTR"], loc="best")
    plt.xlim(min(t), max(t))

    plt.subplot(3, 1, 2)
    plt.plot(t, np.median(T_array, axis=1), "c-", lw=3)
    plt.gca().fill_between(
        t, np.min(T_array, axis=1), np.max(T_array, axis=1), color="c", alpha=0.2
    )
    plt.ylabel("T (K)")
    plt.xlabel("Time (min)")
    plt.legend(["Reactor Temperature"], loc="best")
    plt.xlim(min(t), max(t))

    plt.subplot(3, 1, 3)
    plt.plot(t, tc_trace, "b--", lw=1)
    plt.plot(t, [lower_Tc for _ in range(len(tc_trace))], "r--", lw=1)
    plt.plot(t, [upper_Tc for _ in range(len(tc_trace))], "r--", lw=1)
    plt.ylabel("Cooling T (K)")
    plt.xlabel("Time (min)")
    plt.legend(["Jacket Temperature"], loc="best")
    plt.xlim(min(t), max(t))

    plt.tight_layout()
    plt.show()


def plot_validation_result(
    Ca_trend: list,
    T_trend: list,
    Tc_trend: list,
    ideal_Ca: float,
    ideal_T: float,
    lower_Tc: float,
    upper_Tc: float,
):
    t = range(len(Tc_trend))

    plt.figure(figsize=(8, 5))

    plt.subplot(3, 1, 1)
    plt.plot(t, Ca_trend, "b-o", lw=1, markersize=3)
    plt.plot(t, [ideal_Ca for _ in range(len(Tc_trend))], "r--", lw=1)

    plt.ylabel("Ca (mol/m^3)")
    plt.xlabel("Time (min)")
    plt.legend(["Ca"], loc="best")
    plt.xlim(min(t), max(t))

    plt.subplot(3, 1, 2)
    plt.plot(t, T_trend, "b-o", lw=1, markersize=3)
    plt.plot(t, [ideal_T for _ in range(len(Tc_trend))], "r--", lw=1)

    plt.ylabel("Cooling T (K)")
    plt.xlabel("Time (min)")
    plt.legend(["T"], loc="best")
    plt.xlim(min(t), max(t))

    plt.subplot(3, 1, 3)
    plt.plot(t, Tc_trend, "b-o", lw=1, markersize=3)
    plt.plot(t, [lower_Tc for _ in range(len(Tc_trend))], "r--", lw=1)
    plt.plot(t, [upper_Tc for _ in range(len(Tc_trend))], "r--", lw=1)
    plt.ylabel("Tc (K)")
    plt.xlabel("Time (min)")
    plt.legend(["Tc"], loc="best")
    plt.xlim(min(t), max(t))

    plt.tight_layout()
    plt.show()
