import math
import numpy as np


class OnlineVariableStepLPF:
    def __init__(self, fc,
                 kp=math.nan, kd=0.0,
                 v_max=math.inf, a_max=math.inf):
        """
        Online, variable-step low-pass filter + PID-style limiter:
          • exact-dt one-pole response (3-pole cascade for C² smoothness)
          • holds last input between updates
          • update(t, x) at each new irregular sample
          • sample(t) at any higher-rate time
          • velocity/acceleration limits + PD feedback on error

        Args:
            fc: cutoff frequency (Hz)
            kp: proportional gain (default: use 1/dt → unity tracking)
            kd: derivative gain (damping term)
            v_max: maximum allowed velocity (units/s)
            a_max: maximum allowed acceleration (units/s²)
        """
        self.RC = 1.0 / (2.0 * math.pi * fc)

        # 3-pole cascade states
        self.y1 = 0.0
        self.y2 = 0.0
        self.y3 = 0.0
        self.x_prev = 0.0
        self.t_lpf = 0.0

        # Limiter + PD states
        self.y = 0.0
        self.v = 0.0
        self.t_lim = 0.0
        self.kp = kp
        self.kd = kd
        self.v_max = v_max
        self.a_max = a_max

        # Target derivative states
        self.y_target_prev = 0.0
        self.t_target_prev = 0.0

        self.initialized = False

    def update(self, t, x):
        # print(f"x_prev: {self.x_prev}, v: {self.v}")
        
        """Call when a new (t, x) arrives."""
        if not self.initialized:
            # First update - initialize all values
            self.y1 = x
            self.y2 = x
            self.y3 = x
            self.x_prev = x
            self.t_lpf = t
            self.y = x
            self.t_lim = t
            self.y_target_prev = x
            self.t_target_prev = t
            self.initialized = True
            return

        dt = t - self.t_lpf
        if dt > 0:
            self._step_lpf(dt)
            self.t_lpf = t
        self.x_prev = x

    def sample(self, t):
        """Call at any t ≥ last update time to get smooth, non-overshooting y."""
        # 1) advance LPF
        dt_lpf = t - self.t_lpf
        if dt_lpf > 0:
            self._step_lpf(dt_lpf)
            self.t_lpf = t

        # 2) get new target from LPF
        y_target = self.y3

        # compute target derivative
        dt_tgt = t - self.t_target_prev
        if dt_tgt > 0:
            v_target = (y_target - self.y_target_prev) / dt_tgt
        else:
            v_target = 0.0
        self.y_target_prev = y_target
        self.t_target_prev = t

        # 3) limiter / PD step
        dt_lim = t - self.t_lim
        if dt_lim > 0:
            # error and proportional term
            err = y_target - self.y
            kp_eff = (1.0 / dt_lim) if math.isnan(self.kp) else self.kp
            v_p = kp_eff * err

            # derivative term on error
            v_d = self.kd * (v_target - self.v)

            # desired velocity
            v_des = v_p + v_d

            # limit accel
            dv_max = self.a_max * dt_lim
            dv = max(min(v_des - self.v, dv_max), -dv_max)
            v_new = self.v + dv

            # limit vel
            v_new = max(min(v_new, self.v_max), -self.v_max)

            # step output
            y_new = self.y + v_new * dt_lim

            # commit
            self.v = v_new
            self.y = y_new
            self.t_lim = t

        return self.y

    def _step_lpf(self, dt):
        alpha = math.exp(-dt / self.RC)
        self.y1 = alpha * self.y1 + (1.0 - alpha) * self.x_prev
        self.y2 = alpha * self.y2 + (1.0 - alpha) * self.y1
        self.y3 = alpha * self.y3 + (1.0 - alpha) * self.y2

class MultiDimLPF:
    def __init__(self, dims, **kwargs):
        self.filters = [OnlineVariableStepLPF(**kwargs) for _ in range(dims)]
    
    def reset(self):
        for f in self.filters:
            f.initialized = False
            f.y1 = 0.0
            f.y2 = 0.0
            f.y3 = 0.0
            f.x_prev = 0.0
            f.t_lpf = 0.0
            f.y = 0.0
            f.v = 0.0
            f.t_lim = 0.0
            f.y_target_prev = 0.0
            f.t_target_prev = 0.0
        
    def update(self, t, x):
        for i, f in enumerate(self.filters):
            f.update(t, float(x[i]))

    def sample(self, t):
        return np.array([f.sample(t) for f in self.filters])
