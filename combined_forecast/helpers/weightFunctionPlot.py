import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1) Global figure size
FIG_SIZE = (12, 6)

# 2) Build a time index from 2012-01-01 through 2018-12-31 at 15‑minute intervals (96 per day)
date_idx = pd.date_range(start="2012-01-01",
                         end="2018-12-31 23:45",
                         freq="15T")
# 't' is just an integer counter for computing y
t = np.arange(len(date_idx))


def plotFigure4():
    # Constants
    A, B, C = 100.0, 0.1, 0.0

    # Compute y
    y = A + B * np.cos(2 * np.pi * t / (365 * 96)) + C * t

    # Plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(date_idx, y, label=f"A={A}, B={B}, C={C}")
    ax.set_ylim(0, 200)
    ax.set_ylabel("weights")
    ax.set_xlabel("Year")

    # Year ticks on the x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(date_idx[0], date_idx[-1])

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    # --- your data-loading and combo_counts code stays the same ---
    file_path = "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/WEWResults.csv"
    df = pd.read_csv(file_path)

    combo_counts = (
        df.groupby(['best_A', 'best_B', 'best_C'])
          .size()
          .reset_index(name='count')
          .sort_values(by='count', ascending=False)
    )
    print(combo_counts)
    combo_counts.to_csv("ABC_combination_counts.csv", index=False)
    print("\nSaved combination counts to: ABC_combination_counts.csv")

    # --- bar chart of top 10 combos (same FIG_SIZE) ---
    top_n = 10
    top_combos = combo_counts.head(top_n)
    labels = [f"A={r.best_A}, B={r.best_B}, C={r.best_C}"
              for _, r in top_combos.iterrows()]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.barh(labels, top_combos['count'], color='skyblue')
    ax.set_xlabel('Count')
    ax.invert_yaxis()   # highest at top
    ax.set_ylabel("weights")  # if you want consistency; otherwise remove
    ax.set_title("Top 10 (A, B, C) combinations")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()

    # --- define your function ---
    def func(x, A, B, C):
        raw = A + B * np.cos(2 * np.pi * x / (365 * 96)) + C * x
        return np.maximum(raw, 0)

    # --- plot the top 4 parameter sets ---
    top_4_params = combo_counts.head(4)[['best_A', 'best_B', 'best_C']].values

    for A, B, C in top_4_params:
        y = func(t, A, B, C)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(date_idx, y, label=f"A={A}, B={B}, C={C}")
        ax.set_ylabel("weights")
        ax.set_xlabel("Year")

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(date_idx[0], date_idx[-1])

        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
    plotFigure4()
