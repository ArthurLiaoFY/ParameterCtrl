class PIDController:
    def __init__(
        self,
        Kps: list[float] = [1],
        Kis: list[float] = [1],
        Kds: list[float] = [1],
        Kb: float = 0.0,
    ):
        assert len(Kps) == len(Kis) == len(Kds)
        self.Kps = Kps
        self.Kis = Kis
        self.Kds = Kds
        self.Kb = Kb
        self.prev_error = [0] * len(self.Kps)
        self.integral = [0] * len(self.Kps)

    def compute(self, ideal_ys: list[float], realistic_ys: list[float]):
        output = 0
        for idx, (ideal_y, realistic_y) in enumerate(zip(ideal_ys, realistic_ys)):
            error = ideal_y - realistic_y
            self.integral[idx] += error
            derivative = error - self.prev_error
            output += (
                self.Kps[idx] * error
                + self.Kis[idx] * self.integral[idx]
                + self.Kds[idx] * derivative
            )
            self.prev_error[idx] = error

        return output + self.Kb
