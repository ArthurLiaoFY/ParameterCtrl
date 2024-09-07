import matplotlib.pyplot as plt
import numpy as np


class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float, ideal_y: float, Kb: float = 0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kb = Kb
        self.ideal_y = ideal_y
        self.prev_error = 0
        self.integral = 0

    def compute(self, realistic_y: float):
        error = self.ideal_y - realistic_y
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return output


class LightIntensityEnv:
    def __init__(self, initial_target_intensity):
        self.intensity = initial_target_intensity

    def update(self, power, time):
        noise = np.random.uniform(-3, 6)
        # noise = np.random.uniform(-1.16, 0.2)
        # degradation = 0.01 *  np.random.uniform(0, 0.0412) * time
        degradation = 0.01 * 0.042 * time
        # degradation = 0

        self.intensity = 1.38681004 * power - 12.75095789 + noise - degradation
        return self.intensity


if __name__ == "__main__":
    # 初始參數和設置
    initial_intensity = 45  # 初始光強度
    setpoint = 43  # 目標光強度
    time_steps = 1000
    Kp_opt, Ki_opt, Kd_opt = (
        3 / 11.6 / 1.38681004,
        3 / 11.6 / 1.38681004,
        3 / 11.6 / 1.38681004,
    )
    # Kp_opt, Ki_opt, Kd_opt = 0.0013, 0.422, 0.005

    # 使用最佳參數模擬
    pid_opt = PIDController(Kp=Kp_opt, Ki=Ki_opt, Kd=Kd_opt, ideal_y=setpoint)

    intensity_env = LightIntensityEnv(initial_intensity)

    intensity_opt = []
    power_opt = []

    for i in range(time_steps):
        current_intensity = intensity_env.intensity
        power = pid_opt.compute(current_intensity)
        power = max(35, min(45, power))
        intensity_opt.append(current_intensity)
        power_opt.append(power)
        intensity_env.update(power, i)

    # 繪圖
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    # plt.plot(actual_output, label='Actual Output')
    plt.plot(intensity_opt, label="Optimized Intensity")
    plt.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Time step")
    plt.ylabel("Intensity")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(power_opt, "o", label="Optimized Power")
    plt.xlabel("Time step")
    plt.ylabel("Power")
    plt.legend()

    plt.tight_layout()
    plt.show()
