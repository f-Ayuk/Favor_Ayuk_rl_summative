import os
import pandas as pd
import itertools

from training.dqn_training import train_dqn
from training.pg_training import train_policy


DQN_CONFIGS = [
    {"lr": lr, "gamma": gamma,
     "epsilon_decay": decay, "batch_size": batch,
     "target_update": target}
    for lr, gamma, decay, batch, target in [
        (0.001, 0.99, 0.995, 64, 100),
        (0.0005, 0.99, 0.995, 64, 100),
        (0.002, 0.99, 0.995, 64, 100),
        (0.001, 0.95, 0.995, 64, 100),
        (0.001, 0.999, 0.995, 64, 100),
        (0.001, 0.99, 0.990, 64, 100),
        (0.001, 0.99, 0.999, 64, 100),
        (0.001, 0.99, 0.995, 32, 100),
        (0.001, 0.99, 0.995, 128, 100),
        (0.001, 0.99, 0.995, 64, 250)
    ]
]


POLICY_CONFIGS = [
    {"lr": lr, "gamma": gamma,
     "entropy_coef": entropy,
     "clip_coef": clip,
     "value_coef": value,
     "gae_lambda": gae}
    for lr, gamma, entropy, clip, value, gae in [
        (0.001, 0.99, 0.01, 0.20, 0.5, 0.95),
        (0.0005, 0.99, 0.01, 0.20, 0.5, 0.95),
        (0.002, 0.99, 0.01, 0.20, 0.5, 0.95),
        (0.001, 0.95, 0.01, 0.20, 0.5, 0.95),
        (0.001, 0.999, 0.01, 0.20, 0.5, 0.95),
        (0.001, 0.99, 0.05, 0.20, 0.5, 0.95),
        (0.001, 0.99, 0.001, 0.20, 0.5, 0.95),
        (0.001, 0.99, 0.01, 0.10, 0.5, 0.95),
        (0.001, 0.99, 0.01, 0.30, 1.0, 0.95),
        (0.001, 0.99, 0.01, 0.20, 0.5, 0.80)
    ]
]


def run_experiments(episodes=300):

    os.makedirs("logs", exist_ok=True)

    all_results = []

    algorithms = [
        "DQN",
        "REINFORCE",
        "PPO",
        "A2C"
    ]

    for algorithm in algorithms:

        configs = (
            DQN_CONFIGS
            if algorithm == "DQN"
            else POLICY_CONFIGS.copy()
        )

        for run, config in enumerate(configs, start=1):

            print(
                f"\n{algorithm}: Experiment {run}/10"
            )

            if algorithm == "DQN":

                history = train_dqn(
                    config, episodes=episodes,
                    seed=42
                )

            else:

                history = train_policy(
                    algorithm,
                    config,
                    episodes=episodes,
                    seed=42
                )

            history["algorithm"] = algorithm
            history["experiment"] = run

            for key, value in config.items():
                history[key] = value

            history.to_csv(
                f"logs/{algorithm.lower()}_run_{run}.csv",
                index=False
            )

            all_results.append(history)

    combined = pd.concat(
        all_results, ignore_index=True
    )

    combined.to_csv(
        "logs/all_experiments.csv",
        index=False
    )

    print("\nAll 40 experiments completed.")

    return combined