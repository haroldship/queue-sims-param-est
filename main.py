# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import simpy
import numpy as np


def compute_x(x0, lam, mu, u, G, t):
    x = np.maximum(x0 + lam * t - np.matmul(G, mu * u) * t, 0)
    return x


if __name__ == '__main__':
    x01, x02, x03 = 10, 10, 10
    lam1, lam2, lam3 = lam = np.array((1.0, 1.0, 0.0))
    mu1, mu2, mu3 = mu = np.array((1.0, 1.0, 1.0))
    C1 = 1.0
    G = np.array(((1.0, 0, 0),(0, 1.0, 0),(0, -1.0, 1.0)))
    T = 30

    np.random.seed(234)

    MC = 1

    for i in range(MC):
        for u in range(10):
            # Run the simulation for autoscaled
            env = simpy.Environment()

            x0 = np.array((x01, x02, x03))

            u1 = round(1.0 - u * 0.1, 1)
            u2 = round(C1 - u1, 1)
            u3 = 1.0
            u = np.array((u1, u2, u3))

            for t in range(0, T+1):
                print(f'(u1,u2,u3) = ({(u1, u2, u3)}) x({t}) = {compute_x(x0, lam, mu, u, G, t)}')