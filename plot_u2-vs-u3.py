import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stats_df = pd.read_csv('experiment_u2-vs-u3.csv')

    markers = ['o','+','<','>']

    for i, r in stats_df.iterrows():
        plt.scatter(r.cost_mean, r.mu3_var, marker=markers[i], label=f'$u_2=${r["u2"]} $u_3$={r["u3"]}')

    plt.ylabel(r'Var$(\mu_3)$')
    plt.xlabel('Mean cost')
    plt.legend()
    plt.savefig('experiment_u2-vs-u3.pdf')