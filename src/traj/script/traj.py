import numpy as np

class CircleTrajectory(object):
    def __init__(self, origin, radius=5, omega=0.5):
        self.radius = radius
        self.origin = origin
        self.omega = omega

    def step(self, t):
        x = self.radius*np.cos(self.omega*t) + self.origin[0]
        y = self.radius*np.sin(self.omega*t) + self.origin[1]
        z = self.origin[2]
        return np.array([x,y,z])