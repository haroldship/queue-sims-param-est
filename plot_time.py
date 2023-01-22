import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stats_df = pd.read_csv('experiment_time.csv')

    plt.scatter(stats_df['T'], stats_df.mu3_var)

    plt.ylabel(r'Var$(\mu_3)$')
    plt.xlabel('Time interval')
    plt.savefig('experiment_var-vs-T.pdf')

    plt.clf()

    plt.scatter(stats_df['T'], stats_df.mu3_mean)

    plt.ylabel(r'E$(\mu_3)$')
    plt.xlabel('Time interval')
    plt.savefig('experiment_mean-vs-T.pdf')
