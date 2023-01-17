# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as npr


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
    return sigma_2 * (1 / (u.dot(u) * TT))


df = pd.DataFrame({'e': [], 'mc': [], 'J': [], 'sigma_2': []})


task_df = pd.DataFrame(None, columns=('task_no', 'task_type', 'arrive_time', 'start_time', 'complete_time'))
queue_df = pd.DataFrame(None, columns=('task_type', 'time', 'length'))


def process_task(env, service_time):
    yield env.timeout(service_time)


def enter_network(env, task_no, buffers, task_type, mu):
    arrive_time = env.now
    task_index = task_type - 1

    queue = buffers[task_index]
    queue_df.loc[queue_df.size] = (task_type, arrive_time, len(queue.queue))
    with queue.request() as req:
        yield req
        start_time = env.now
        queue_df.loc[queue_df.size] = (task_type, start_time, len(queue.queue))
        service_time = npr.exponential(1.0/mu[task_index])
        complete_time = start_time + service_time
        task_df.loc[task_df.size] = (task_no, task_type, arrive_time, start_time, complete_time)
        yield env.process(process_task(env, service_time))
        queue_df.loc[queue_df.size] = (task_type, complete_time, len(queue.queue))


def run_network(env, buffers, x0, lam, mu):
    task_no = 0
    total_rate = np.sum(lam)
    probs = lam / total_rate
    ntypes = len(lam)
    for k in range(ntypes):
        for n in range(x0[k]):
            task_type = k + 1
            task_no += 1
            env.process(enter_network(env, task_no, buffers, task_type, mu))

    while True:
        yield env.timeout(npr.exponential(1.0 / total_rate))
        task_type = npr.choice(ntypes, p=probs) + 1
        task_no += 1
        env.process(enter_network(env, task_no, buffers, task_type, mu))


def run_random_arrivals():
    npr.seed(1234)

    x01, x02, x03 = x0 = np.array((10, 10, 10))
    lam1, lam2, lam3 = lam = np.array((1.0, 1.0, 0.0))
    mu1, mu2, mu3 = mu = np.array((1.0, 1.0, 1.0))
    u1, u2, u3 = u = np.array((1.0, 1.0, 1.0))
    C1, C2 = C = np.array((1.0, 1.0)) # capacity of servers 1, 2
    c = np.array((1.0, 1.0, 1.0)) # hold cost per item per unit time
    G = np.array(((1.0, 0, 0),(0, 1.0, 0),(0, -1.0, 1.0)))
    TT = 30

    env = simpy.Environment()

    x1, x2, x3, = buffers = [simpy.Resource(env, capacity=1) for i in range(3)]

    env.process(run_network(env, buffers, x0, lam, mu))
    env.run(TT)
    print(task_df)
    print(queue_df)


def run_random_controls():
    x01, x02, x03 = 10, 10, 10
    lam1, lam2, lam3 = lam = np.array((1.0, 1.0, 0.0))
    C1 = 1.0 # capacity of server 1
    c = np.array((1.0, 1.0, 1.0))
    G = np.array(((1.0, 0, 0),(0, 1.0, 0),(0, -1.0, 1.0)))
    TT = 30

    np.random.seed(1234)
    mu_mid = 1.0
    mu_wid = 0.25
    sigma_2 = mu_wid**2 / 12.0
    Sigma_2 = np.eye(3) * sigma_2

    MC = 100
    u3s = []

    for mc in range(MC):
        for e in np.arange(0, 1.1, 0.1):
            # Run the simulation in an environment
            # env = simpy.Environment()

            mu1, mu2, mu3 = mu = np.random.uniform(mu_mid - mu_wid / 2.0, mu_mid + mu_wid / 2.0, 3)

            x0 = np.array((x01, x02, x03))

            u1 = 1.0 - e
            u2 = e
            u3 = np.random.uniform(0.0, 1.0)
            # if u1 + u2 > 1.0:
            #     u1 = round(u1/(u1+u2), 2)
            #     u2 = round(1.0 - u2, 2)
            u = np.array((u1, u2, u3))

            assert all(u >= 0), f"u={u}"
            assert all(u <= 1.0), f"u={u}"

            df.loc[len(df)] = [e, mc, compute_cost(c, x0, lam, mu, u, G, TT), compute_variance(Sigma_2, u, TT)[2,2]]
            u3s.append(u3)

    dfm = df.groupby('e')[['J', 'sigma_2']].mean()

    plt.scatter(dfm.J, dfm.sigma_2)
    plt.xlabel('J')
    plt.title(r'Var($\hat{\beta}_3(u)$) vs $J(u)$')
    plt.savefig('beta_hat-vs-J.pdf')
    plt.cla()
    plt.scatter(range(len(u3s)), u3s)
    plt.title(r'$u_3$ in simulations')
    plt.savefig('u3_values.pdf')


if __name__ == '__main__':
    run_random_arrivals()
