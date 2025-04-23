import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None
        self.step = 0

    def transform(self, X):
        if self.step == 0:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            self.step += X.shape[0]

        else:
            new_min = np.min(X, axis=0)
            new_max = np.max(X, axis=0)
            self.min = np.minimum(self.min, new_min)
            self.max = np.maximum(self.max, new_max)
            self.step += X.shape[0]

        return (X - self.min) / (self.max - self.min)


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.S = None
        self.step = 0

    def transform(self, X):
        if self.step == 0:
            self.step += X.shape[0]
            self.mean = np.mean(X, axis=0)
            self.S = self.step * np.var(X, axis=0)
            std = np.sqrt(self.S / self.step)

        else:
            batch_size = X.shape[0]
            new_step = self.step + batch_size
            new_mean = np.mean(X, axis=0)
            self.mean = (self.mean * self.step + new_mean * batch_size) / new_step
            self.S = self.S + (self.step * batch_size / new_step) * (new_mean - self.mean) ** 2
            std = np.sqrt(self.S / self.step)
            self.step = new_step

        return (X - self.mean) / std
