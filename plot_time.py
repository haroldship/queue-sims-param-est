import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stats_df = pd.read_csv('experiment_time.csv')

    plt.scatter(stats_df['T'], stats_df.mu3_var, label=r'Var$(\hat{\mu}_3)$')
    plt.plot(stats_df['T'], 2.0/stats_df['T'], c='red', linestyle='dotted', label=r'$y=2/t$')
    plt.legend()

    plt.ylabel(r'Var$(\hat{\mu}_3)$')
    plt.xlabel('Time interval')
    plt.savefig('experiment_var-vs-T.pdf')

    plt.clf()

    plt.scatter(stats_df['T'], stats_df.mu3_mean)

    plt.ylabel(r'E$(\hat{\mu}_3)$')
    plt.xlabel('Time interval')
    plt.savefig('experiment_mean-vs-T.pdf')
