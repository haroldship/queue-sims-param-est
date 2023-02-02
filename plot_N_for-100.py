import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stats_df = pd.read_csv('experiment_sample-size-for-100.csv')

    plt.scatter(stats_df.N, stats_df.mu3_var, label=r'Var$(\hat{\mu}_3)$')
    #plt.plot(stats_df['T'], 2.0/stats_df['T'], c='red', linestyle='dotted', label=r'$y=2/t$')
    plt.ylim(0, 1e-4)
    #plt.legend()

    plt.ylabel(r'Var$(\hat{\mu}_3)$')
    plt.xlabel('Sample Size')
    plt.savefig('experiment_var-vs-N-for-100.pdf')

    plt.clf()

    plt.scatter(stats_df.N, stats_df.mu3_mean)

    plt.ylabel(r'E$(\hat{\mu}_3)$')
    plt.xlabel('Sample size')
    plt.savefig('experiment_mean-vs-N-for-100.pdf')
