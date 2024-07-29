from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    An enhanced PID controller with adaptive gains, anti-windup, and smoothing
    """
    def __init__(self):
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.max_integral = 10.0  # limit for antiwindup
        self.error_integral = 0
        self.prev_error = 0
        self.prev_derivative = 0  # init the prev derivative for smoothing
        self.filter_coefficient = 0.9  # Smoothing factor for derivative term

        self.adaptive_gain = False

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = max(min(self.error_integral, self.max_integral), -self.max_integral)  # Anti-windup

        error_diff = error - self.prev_error
        smoothed_diff = self.filter_coefficient * self.prev_derivative + (1 - self.filter_coefficient) * error_diff  
        self.prev_derivative = smoothed_diff  

        self.prev_error = error

        if self.adaptive_gain:
            if state.v_ego > 30:  
                self.p = 0.4
                self.i = 0.07
            else:
                self.p = 0.3
                self.i = 0.05

        feedforward = 0
        if future_plan is not None and future_plan.lataccel:
            if len(future_plan.lataccel) > 0:
                feedforward = 0.1 * future_plan.lataccel[0]  

        return self.p * error + self.i * self.error_integral + self.d * smoothed_diff + feedforward
