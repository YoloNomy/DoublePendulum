import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Pendule():
    def __init__(l1, l2, m1, m2):
        # Parameters
        self.l1 = l1,
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

        # Adim parameters
        self.gamma = l2 / l1
        self.beta = m2 / (m1 + m2)

    def Fevol(t, y):
        """Evolution function.
        """
        q1, q2, p1, p2 = y[0], y[1], y[2], y[3]
        dq1 = p1
        dq2 = p2
        dtheta = q1 - q2
        D_P1 = 1 - self.beta * np.cos(dthea)**2
        D_P2 = self.gamma * (self.beta * np.cos(dtheta)**2 - 1)
        N_P1 = self.beta * (np.cos(dtheta) * (np.sin(q2)
        dp1 = (beta*np.cos(q1-q2)*np.sin(q2) - np.sin(q1) - beta*np.sin(q1-q2)*(gamma*p2**2 + np.cos(q1-q2)*p1**2))/(1-(beta*np.cos(q1-q2)*np.cos(q1-q2)))
        dp2 = (np.sin(q2) - np.cos(q1-q2)*np.sin(q1) - np.sin(q1-q2)*(p1**2 + beta*gamma*np.cos(q1-q2)*p2**2)) / (beta*gamma*np.cos(q1-q2)*np.cos(q1-q2) - gamma)
    return(dq1, dq2, dp1, dp2)

    def run_pendulum(q1, q2, p1=0, p2=0, tmax):
        y0 = np.array([np.radians(q1), np.radians(q2), p1, p2])
        sol = solve_ivp(Fevol, (0, tmax))
        theta1 = sol[:,0]
        theta2 = sol[:,1]
