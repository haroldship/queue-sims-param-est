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


def enter_network(env, G, buffers, task_type, mu, skip_queue=False):
    arrive_time = env.now
    task_index = task_type - 1

    queue = buffers[task_index]
    if not skip_queue:
        queue_df.loc[queue_df.size] = (task_type, arrive_time, len(queue.queue))
    with queue.request() as req:
        yield req
        start_time = env.now
        if not skip_queue:
            queue_df.loc[queue_df.size] = (task_type, start_time, len(queue.queue))
        service_time = npr.exponential(1.0/mu[task_index])
        complete_time = start_time + service_time
        yield env.process(process_task(env, service_time))
        if not skip_queue:
            queue_df.loc[queue_df.size] = (task_type, complete_time, len(queue.queue))
        task_no = task_df.size
        task_df.loc[task_no] = (task_no, task_type, arrive_time, start_time, complete_time)
        J = G.shape[1]
        for j in range(J):
            if j == task_index: continue
            p_kj = -G[j, task_index]
            if npr.binomial(1, p_kj):
                env.process(enter_network(env, G, buffers, j+1, mu))


def run_network(env, G, buffers, x0, lam, mu):
    task_no = 0
    total_rate = np.sum(lam)
    probs = lam / total_rate
    ntypes = len(lam)
    for k in range(ntypes):
        for n in range(x0[k]):
            task_type = k + 1
            queue_df.loc[queue_df.size] = (task_type, 0, x0[k])
            task_no += 1
            env.process(enter_network(env, G, buffers, task_type, mu, skip_queue=True))

    while True:
        yield env.timeout(npr.exponential(1.0 / total_rate))
        task_type = npr.choice(ntypes, p=probs) + 1
        task_no += 1
        env.process(enter_network(env, G, buffers, task_type, mu))


def q(k, t):
    queue_df_k = queue_df[(queue_df.task_type == k)]
    queue_df_kt = queue_df_k[(queue_df_k.time >= t)]
    if queue_df_kt.size == 0:
        if t < 1e-5:
            return queue_df_k.iloc[0]['length']
        else:
            return queue_df_k.iloc[-1]['length']
    return queue_df_kt.iloc[0]['length']


def run_random_arrivals():
    npr.seed(1)

    x01, x02, x03 = x0 = np.array((10, 10, 10))
    lam1, lam2, lam3 = lam = np.array((1.0, 1.0, 0.0))
    mu1, mu2, mu3 = mu = np.array((1.0, 1.0, 1.0))
    u1, u2, u3 = u = np.array((1.0, 1.0, 1.0))
    C1, C2 = C = np.array((1.0, 1.0)) # capacity of servers 1, 2
    c = np.array((1.0, 1.0, 1.0)) # hold cost per item per unit time
    G = np.array(((1.0, 0, 0),(0, 1.0, 0),(0, -1.0, 1.0)))
    TT = 30
    dt = 0.5

    env = simpy.Environment()

    x1, x2, x3, = buffers = [simpy.Resource(env, capacity=1) for i in range(3)]

    env.process(run_network(env, G, buffers, x0, lam, mu))
    env.run(TT)
    task_df['wait_time'] = task_df.start_time - task_df.arrive_time
    task_df['service_time'] = task_df.complete_time - task_df.start_time
    task_df['sojourn_time'] = task_df.complete_time - task_df.arrive_time

    sample_times = np.arange(0, TT + dt, dt)
    for t in sample_times:
        print(f'q3({t})={q(3,t)}')

    X_t = np.array([u3 * t for t in sample_times]).reshape(len(sample_times), 1)
    Y_t = np.array([-q(3,t) + x03 + (lam[2] + u2) * t for t in sample_times]).reshape(len(sample_times), 1)

    mu3_hat = np.linalg.inv(X_t.T.dot(X_t)).dot(X_t.T).dot(Y_t)
    print(mu3_hat)

    plt.scatter(X_t, Y_t)
    plt.show()


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
