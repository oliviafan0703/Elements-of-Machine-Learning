import functools
import numpy as np


class CallCounter:
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f
        self.count = 0

    def __call__(self, *args, **kwargs):
        values = self.f(*args, **kwargs)
        self.count += values.size
        return values

    def calls(self):
        return self.count

    def reset(self):
        self.count = 0


class HistoryTracker:
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f
        self.z_history = []
        self.f_history = []

    def __call__(self, *args, **kwargs):
        values = self.f(*args, **kwargs)
        self.z_history.append(values[0])
        self.f_history.append(values[1])
        return values

    def history(self):
        return {'z': np.array(self.z_history),
                'f': np.array(self.f_history)}

    def restart(self, z0, f0):
        self.z_history = [z0]
        self.f_history = [f0]
