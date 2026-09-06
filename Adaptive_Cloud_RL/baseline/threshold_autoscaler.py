class ThresholdAutoscaler:
    """Measured busy-time utilization, hysteresis, cooldown and low-load streak."""
    name = "Threshold baseline"

    def __init__(self, scale_out_threshold=.75, scale_in_threshold=.30,
                 cooldown_steps=1, scale_in_stabilization_steps=2):
        if not 0 <= scale_in_threshold < scale_out_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= in < out <= 1")
        if cooldown_steps < 0 or scale_in_stabilization_steps < 1:
            raise ValueError("Invalid cooldown or stabilization period")
        self.high, self.low = scale_out_threshold, scale_in_threshold
        self.cooldown_steps, self.stabilization = cooldown_steps, scale_in_stabilization_steps
        self.reset()

    def reset(self):
        self.cooldown = self.low_streak = 0

    def select(self, info):
        mask, utilization = info["action_mask"], info["utilization"]
        self.low_streak = self.low_streak+1 if utilization < self.low else 0
        if self.cooldown:
            self.cooldown -= 1
            return 1
        action = 1
        if utilization > self.high and mask[2]:
            action = 2
        elif self.low_streak >= self.stabilization and mask[0]:
            action = 0
        if action != 1:
            self.cooldown = self.cooldown_steps
            self.low_streak = 0
        return action

