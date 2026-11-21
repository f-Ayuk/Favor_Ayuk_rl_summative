import os
import pandas as pd
import matplotlib.pyplot as plt

from training.experiments import run_experiments


def plot_results(results):

    algorithms = results["algorithm"].unique()

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9)
    )

    axes = axes.flatten()

    for i, algorithm in enumerate(algorithms):

        subset = results[
            results["algorithm"] == algorithm
        ]

        for experiment, group in subset.groupby(
            "experiment"
        ):

            rolling = group["reward"].rolling(
                20, min_periods=1
            ).mean()

            axes[i].plot(
                group["episode"],
                rolling,
                alpha=0.65,
                label=f"Run {experiment}"
            )

        axes[i].set_title(algorithm)
        axes[i].set_xlabel("Episode")
        axes[i].set_ylabel("Reward")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)

    plt.savefig(
        "assets/learning_curves.png",
        dpi=200
    )

    plt.show()


def main():

    print("=" * 60)
    print("SMART IRRIGATION RL CAPSTONE")
    print("=" * 60)

    results = run_experiments(episodes=300)

    plot_results(results)

    summary = (
        results.groupby(
            ["algorithm", "experiment"]
        )
        .agg(
            mean_reward=("reward", "mean"),
            final_health=("health", "last"),
            water_used=("water", "mean"),
            energy_used=("energy", "mean")
        )
        .reset_index()
    )

    summary.to_csv(
        "logs/experiment_summary.csv",
        index=False
    )

    print("\nExperiment summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()