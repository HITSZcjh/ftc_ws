import numpy as np

class CircleTrajectory(object):
    def __init__(self, origin, radius=5, omega=0.5):
        self.radius = radius
        self.origin = origin
        self.omega = omega
        self.rate = 0
    def step(self, t, cnt):
        self.rate = np.clip(cnt*0.005, 0, 1)
        x = self.radius*np.cos(self.rate*self.omega*t) + self.origin[0]
        y = self.radius*np.sin(self.rate*self.omega*t) + self.origin[1]
        z = self.origin[2]
        return np.array([x,y,z])