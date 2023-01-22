# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import simpy
import numpy as np
import pandas as pd
import numpy.random as npr
import statistics
import datetime

def compute_cost(queue_df, c, TT, sample_times):
    cost = 0
    K = len(c)
    dt = TT / len(sample_times)
    for t in sample_times:
        x_t = np.array([q(queue_df, k+1, t) for k in range(K)])
        cost += c.dot(x_t) * dt
    return cost


def q(queue_df, k, t):
    queue_df_k = queue_df[(queue_df.task_type == k)]
    queue_df_kt = queue_df_k[(queue_df_k.time >= t)]
    if queue_df_kt.size == 0:
        if t < 1e-5:
            return queue_df_k.iloc[0]['length']
        else:
            return queue_df_k.iloc[-1]['length']
    return queue_df_kt.iloc[0]['length']


def estimate_one_rate(queue_df, x0, lam, mu, u, G, sample_times, task_type):
    task_index = task_type - 1
    X_t = np.array([u[task_index] * t for t in sample_times]).reshape(len(sample_times), 1)
    Y_t = np.array([-q(queue_df, task_type, t) + x0[task_index] + (lam[task_index] - sum([u[j] * G[task_index, j] * mu[j] for j in range(G.shape[1]) if G[task_index, j] < 0])) * t for t in sample_times]).reshape(len(sample_times), 1)
    slope = np.linalg.inv(X_t.T.dot(X_t)).dot(X_t.T).dot(Y_t)
    return slope[0,0]


def process_task(env, service_time):
    yield env.timeout(service_time)


def enter_network(task_df, queue_df, env, G, buffers, task_type, mu, u, skip_queue=False):
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
        if u[task_index] < 1e-10: yield env.timeout(float('inf'))
        service_time = npr.exponential(1.0/(u[task_index] * mu[task_index]))
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
                env.process(enter_network(task_df, queue_df, env, G, buffers, j+1, mu, u))


def run_network(task_df, queue_df, env, G, buffers, x0, lam, mu, u):
    task_no = 0
    total_rate = np.sum(lam)
    probs = lam / total_rate
    ntypes = len(lam)
    for k in range(ntypes):
        for n in range(x0[k]):
            task_type = k + 1
            queue_df.loc[queue_df.size] = (task_type, 0, x0[k])
            task_no += 1
            env.process(enter_network(task_df, queue_df, env, G, buffers, task_type, mu, u, skip_queue=True))

    while True:
        yield env.timeout(npr.exponential(1.0 / total_rate))
        task_type = npr.choice(ntypes, p=probs) + 1
        task_no += 1
        env.process(enter_network(task_df, queue_df, env, G, buffers, task_type, mu, u))


def run_random_arrivals(name, experiments, MC, network_params):

    x0 = network_params['x0']
    lam = network_params['lam']
    mu = network_params['mu']
    C = network_params['C']
    c = network_params['c']
    G = network_params['G']

    stats_df = pd.DataFrame(None, columns=('T', 'N', 'u1', 'u2', 'u3', 'cost_mean', 'mu3_mean', 'mu3_var'), dtype='float')

    for TT, n, u in experiments:

        npr.seed(1)

        costs = []
        mu3_hats = []

        for mc in range(MC):
            if mc % 10 == 9:
                print('.', sep='',end='')

            task_df = pd.DataFrame(None, columns=('task_no', 'task_type', 'arrive_time', 'start_time', 'complete_time'))
            queue_df = pd.DataFrame(None, columns=('task_type', 'time', 'length'))
            
            dt = TT / n
            u = np.array(u)

            env = simpy.Environment()

            buffers = [simpy.Resource(env, capacity=1) for i in range(3)]

            env.process(run_network(task_df, queue_df, env, G, buffers, x0, lam, mu, u))
            env.run(TT)
            task_df['wait_time'] = task_df.start_time - task_df.arrive_time
            task_df['service_time'] = task_df.complete_time - task_df.start_time
            task_df['sojourn_time'] = task_df.complete_time - task_df.arrive_time

            sample_times = np.arange(0, TT + dt, dt)
            mu3_hat = estimate_one_rate(queue_df, x0, lam, mu, u, G, sample_times, 3)
            cost = compute_cost(queue_df, c, TT, sample_times)

            mu3_hats.append(mu3_hat)
            costs.append(cost)

        print()

        cost_mean = statistics.mean(costs)
        mu3_hat_mean = statistics.mean(mu3_hats)
        mu3_hat_var = statistics.variance(mu3_hats, mu3_hat_mean)

        experiment_no = len(stats_df)
        stats_df.loc[experiment_no] = (TT, n, *u, cost_mean, mu3_hat_mean, mu3_hat_var)

    now = datetime.datetime.now()
    if not name:
        name = now.strftime('%Y-%m-%d-%H%M%S')
    stats_df.to_csv(f'experiment_{name}.csv', index=False)


if __name__ == '__main__':
    name = 'sample-size'
    MC = 50
    experiments = [(120, N, (1.0, 1.0, 1.0)) for N in range(10, 250, 10)]
    network_params = {
        'x0': np.array((10, 10, 10)),
        'lam': np.array((1.0, 1.0, 0.0)),
        'mu': np.array((1.0, 1.0, 1.0)),
        'C': np.array((1.0, 1.0)), # capacity of servers 1, 2
        'c': np.array((1.0, 1.0, 1.0)), # hold cost per item per unit time
        'G': np.array(((1.0, 0, 0),(0, 1.0, 0),(0, -1.0, 1.0)))
    }
    run_random_arrivals(name, experiments, MC, network_params)
