import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stats_df = pd.read_csv('experiment_u2-vs-u3.csv')

    names = [f'$u_2=${r["u2"]} $u_3$={r["u3"]}' for i, r in stats_df.iterrows()]
    types = range(len(names))
    markers = ['o','+','<','>']

    for i, r in stats_df.iterrows():
        plt.scatter(r.mu3_var, r.cost_mean, marker=markers[i], label=names[i])

    plt.xlabel(r'Var$(\mu_3)$')
    plt.ylabel('Mean cost')
    plt.legend()
    plt.savefig('experiment_u2-vs-u3.pdf')