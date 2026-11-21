import os
import numpy as np
import pandas as pd
import torch

from environment.custom_env import SmartIrrigationEnv
from training.algorithms import PolicyAgent, DEVICE


def train_policy(
    algorithm,
    config,
    episodes=300,
    seed=42
):

    env = SmartIrrigationEnv(seed=seed)

    agent = PolicyAgent(
        algorithm=algorithm,
        **config
    )

    history = []

    for episode in range(episodes):

        state, _ = env.reset(seed=seed + episode)

        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "old_log_probs": []
        }

        total_reward = 0
        done = False

        while not done:

            dist, _ = agent.distribution(state)

            action_tensor = dist.sample()
            action = int(action_tensor.item())

            old_log_prob = float(
                dist.log_prob(action_tensor).item()
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(float(done))
            trajectory["old_log_probs"].append(old_log_prob)

            total_reward += reward
            state = next_state

        loss = agent.update(trajectory)

        history.append({
            "episode": episode,
            "reward": total_reward,
            "health": float(np.mean(env.health)),
            "water": env.total_water,
            "energy": env.total_energy,
            "loss": loss
        })

        if episode % 25 == 0:
            print(
                f"{algorithm} Episode {episode}: "
                f"Reward={total_reward:.2f}"
            )

    folder = "models/pg"
    os.makedirs(folder, exist_ok=True)

    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict()
        },
        f"{folder}/{algorithm.lower()}_final.pt"
    )

    return pd.DataFrame(history)