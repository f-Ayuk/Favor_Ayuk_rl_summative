import os
import numpy as np
import pandas as pd
import torch

from environment.custom_env import SmartIrrigationEnv
from training.algorithms import DQNAgent


def train_dqn(config, episodes=300, seed=42):

    env = SmartIrrigationEnv(seed=seed)

    agent = DQNAgent(**config)

    results = []

    for episode in range(episodes):

        state, _ = env.reset(seed=seed + episode)

        total_reward = 0
        losses = []

        done = False

        while not done:

            action = agent.act(state)

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            agent.remember(
                state, action, reward, next_state, done
            )

            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        results.append({
            "episode": episode,
            "reward": total_reward,
            "health": np.mean(env.health),
            "water": env.total_water,
            "energy": env.total_energy,
            "loss": np.mean(losses) if losses else 0
        })

        if episode % 25 == 0:
            print(
                f"DQN Episode {episode}: "
                f"Reward={total_reward:.2f}"
            )

    os.makedirs("models/dqn", exist_ok=True)
    torch.save(
        agent.policy.state_dict(),
        "models/dqn/final.pt"
    )

    return pd.DataFrame(results)


def evaluate_dqn(agent, episodes=20, seed=10000):

    env = SmartIrrigationEnv()
    results = []

    for ep in range(episodes):

        state, _ = env.reset(seed=seed + ep)

        done = False
        total_reward = 0

        while not done:

            action = agent.act(state, greedy=True)

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated
            total_reward += reward

        results.append({
            "reward": total_reward,
            "health": float(np.mean(env.health)),
            "water": env.total_water,
            "energy": env.total_energy
        })

    return pd.DataFrame(results)