import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stats_df = pd.read_csv('experiment_sample-size.csv')

    plt.scatter(stats_df.N, stats_df.mu3_var)

    plt.ylabel(r'Var$(\mu_3)$')
    plt.xlabel('Sample Size')
    plt.savefig('experiment_var-vs-N.pdf')

    plt.clf()

    plt.scatter(stats_df.N, stats_df.mu3_mean)

    plt.ylabel(r'E$(\mu_3)$')
    plt.xlabel('Sample size')
    plt.savefig('experiment_mean-vs-N.pdf')
