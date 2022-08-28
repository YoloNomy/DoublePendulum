import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Pendule():
    def __init__(self, l1, l2, m1, m2):
        # Parameters
        self.l1 = l1,
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

        # Adim parameters
        self.gamma = l2 / l1
        self.beta = m2 / (m1 + m2)

    def Fevol(self, t, y):
        """Evolution function.
        """
        q1, q2, p1, p2 = y[0], y[1], y[2], y[3]
        dq1 = p1
        dq2 = p2

        dtheta = q1 - q2
        D_P1 = 1 - self.beta * np.cos(dtheta)**2
        D_P2 = self.gamma * (self.beta * np.cos(dtheta)**2 - 1)
        N_P1 = self.beta * (np.cos(dtheta) * (np.sin(q2) - p1**2 * np.sin(dtheta))
                            - self.gamma * np.sin(dtheta) * p2**2) - np.sin(q1)
        N_P2 = -np.sin(dtheta) * (p1**2
                                  + self.beta * self.gamma * np.cos(dtheta) * p2**2) + np.sin(q2) - np.cos(dtheta) * np.sin(q1)
        dp1 = N_P1 / D_P1
        dp2 = N_P2 / D_P2

        return dq1, dq2, dp1, dp2

    def run_pendulum(self, q1, q2, p1=0, p2=0, tmax=20):
        y0 = np.array([np.radians(q1), np.radians(q2), p1, p2])
        t, sol = solve_ivp(self.Fevol, (0, tmax), y0)
        theta1 = sol[:, 0]
        theta2 = sol[:, 1]

        self.time = t
        self.theta1 = theta1
        self.theta2 = theta2

    def
