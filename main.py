# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_x(x0, lam, mu, u, G, t):
    x = np.maximum(x0 + lam * t - np.matmul(G, mu * u) * t, 0)
    return x


def compute_cost(c, x0, lam, mu, u, G, TT, dt=0.1):
    J = 0
    for t in np.arange(0.0, TT+dt, dt):
        x_t = compute_x(x0, lam, mu, u, G, t)
        J += c.dot(x_t)
    return J


def compute_variance(sigma_2, u, TT):
    return sigma_2 / (u.dot(u) * TT)


df = pd.DataFrame({'e': [], 'mc': [], 'J': [], 'sigma_2': []})


if __name__ == '__main__':
    x01, x02, x03 = 10, 10, 10
    lam1, lam2, lam3 = lam = np.array((1.0, 1.0, 0.0))
    C1 = 1.0 # capacity of server 1
    c = np.array((1.0, 1.0, 1.0))
    G = np.array(((1.0, 0, 0),(0, 1.0, 0),(0, -1.0, 1.0)))
    TT = 30

    np.random.seed(1234)
    mu_mid = 1.0
    mu_wid = 0.25
    mu1, mu2, mu3 = mu = np.random.uniform(mu_mid - mu_wid/2.0, mu_mid + mu_wid / 2.0, 3)
    sigma_2 = mu_wid**2 / 12.0

    MC = 100
    u3s = []

    for mc in range(MC):
        for e in range(10+1):
            # Run the simulation for autoscaled
            env = simpy.Environment()

            x0 = np.array((x01, x02, x03))

            u1 = round(1.0 - e * 0.1, 1)
            u2 = round(C1 - u1, 1)
            u3 = np.random.uniform(0.0, 1.0)
            u = np.array((u1, u2, u3))

            assert all(u >= 0)
            assert all(u <= 1.0)

            df.loc[len(df)] = [e, mc, compute_cost(c, x0, lam, mu, u, G, TT), compute_variance(sigma_2, u, TT)]
            u3s.append(u3)

    dfm = df.groupby('e')[['J', 'sigma_2']].mean()

    plt.scatter(dfm.J, dfm.sigma_2)
    plt.xlabel('J')
    plt.title(r'Var($\hat{\beta}(u)$) vs $J(u)$')
    plt.savefig('beta_hat-vs-J.pdf')
    plt.cla()
    plt.scatter(range(len(u3s)), u3s)
    plt.title(r'$u_3$ in simulations')
    plt.savefig('u3_values.pdf')
