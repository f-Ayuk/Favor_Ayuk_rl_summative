import argparse
import os
import csv

import numpy as np
import pygame
import torch

from environment.custom_env import SmartIrrigationEnv
from environment.rendering import FarmRenderer
from training.algorithms import DQNAgent, PolicyAgent


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class IrrigationMission:

    def __init__(
        self,
        algorithm="DQN",
        episodes=1,
        seed=42,
        speed=0.15,
        output="logs/mission_results.csv"
    ):

        self.algorithm = algorithm.upper()
        self.episodes = episodes
        self.seed = seed
        self.speed = speed
        self.output = output

        self.env = SmartIrrigationEnv(
            max_days=60
        )

        self.renderer = FarmRenderer()

        self.running = True
        self.paused = False
        self.restart = False

        self.episode = 0
        self.results = []

        self.agent = None

        self.load_agent()

    # -----------------------------------------
    # LOAD TRAINED AGENT
    # -----------------------------------------

    def load_agent(self):

        if self.algorithm == "DQN":

            checkpoint = "models/dqn/final.pt"

            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Missing DQN checkpoint: {checkpoint}"
                )

            self.agent = DQNAgent()

            weights = torch.load(
                checkpoint,
                map_location=DEVICE,
                weights_only=True
            )

            self.agent.policy.load_state_dict(weights)
            self.agent.policy.eval()

            self.agent.epsilon = 0.0

        elif self.algorithm in [
            "REINFORCE", "PPO", "A2C"
        ]:

            checkpoint = (
                f"models/pg/{self.algorithm.lower()}_final.pt"
            )

            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint}"
                )

            self.agent = PolicyAgent(
                algorithm=self.algorithm
            )

            weights = torch.load(
                checkpoint,
                map_location=DEVICE,
                weights_only=True
            )

            self.agent.actor.load_state_dict(
                weights["actor"]
            )

            self.agent.critic.load_state_dict(
                weights["critic"]
            )

            self.agent.actor.eval()
            self.agent.critic.eval()

        else:
            raise ValueError(
                f"Unsupported algorithm: {self.algorithm}"
            )

        print(f"\nLoaded {self.algorithm} successfully.")
        print(f"Device: {DEVICE}")

    # -----------------------------------------
    # AGENT ACTION
    # -----------------------------------------

    def select_action(self, observation):

        with torch.no_grad():

            return self.agent.act(
                observation,
                greedy=True
            )

    # -----------------------------------------
    # ACTION DESCRIPTION
    # -----------------------------------------

    @staticmethod
    def decode_action(action):

        if action == 0:
            return "Pump OFF"

        zone = (action - 1) // 3 + 1
        level = (action - 1) % 3 + 1

        level_names = {
            1: "LOW",
            2: "MEDIUM",
            3: "HIGH"
        }

        return f"Zone {zone} - {level_names[level]}"

    # -----------------------------------------
    # KEYBOARD CONTROLS
    # -----------------------------------------

    def handle_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.restart = True

    # -----------------------------------------
    # SAVE RESULTS
    # -----------------------------------------

    def save_results(self):

        if not self.results:
            return

        os.makedirs(
            os.path.dirname(self.output) or ".",
            exist_ok=True
        )

        fields = [
            "algorithm",
            "episode",
            "reward",
            "crop_health",
            "soil_moisture",
            "water_used",
            "energy_used",
            "reservoir_remaining",
            "days_completed"
        ]

        file_exists = os.path.isfile(self.output)

        with open(
            self.output,
            "a",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            if not file_exists:
                writer.writeheader()

            writer.writerows(self.results)

        print(f"\nResults saved: {self.output}")

    # -----------------------------------------
    # RUN ONE MISSION
    # -----------------------------------------

    def run_episode(self):

        observation, _ = self.env.reset(
            seed=self.seed + self.episode
        )

        total_reward = 0.0
        done = False

        while self.running and not done:

            self.handle_events()

            if not self.running:
                break

            if self.restart:
                return

            if self.paused:

                self.renderer.clock.tick(30)
                continue

            # Agent chooses irrigation action.
            action = self.select_action(observation)

            # Environment executes action.
            (
                next_observation,
                reward,
                terminated,
                truncated,
                _
            ) = self.env.step(action)

            done = terminated or truncated

            total_reward += reward

            observation = next_observation

            self.renderer.draw(self.env)

            if self.speed > 0:
                pygame.time.delay(
                    int(self.speed * 1000)
                )

        if not self.running or self.restart:
            return

        # Mission summary.
        result = {
            "algorithm": self.algorithm,
            "episode": self.episode + 1,
            "reward": round(total_reward, 3),
            "crop_health": round(
                float(np.mean(self.env.health)), 4
            ),
            "soil_moisture": round(
                float(np.mean(self.env.moisture)), 4
            ),
            "water_used": round(
                float(self.env.total_water), 3
            ),
            "energy_used": round(
                float(self.env.total_energy), 3
            ),
            "reservoir_remaining": round(
                float(self.env.reservoir), 4
            ),
            "days_completed": self.env.day
        }

        self.results.append(result)

        print("\n" + "=" * 45)
        print("AGRICULTURAL MISSION COMPLETED")
        print("=" * 45)

        for key, value in result.items():
            print(f"{key}: {value}")

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------

    def run(self):

        print("\nSMART IRRIGATION RL SIMULATOR")
        print("=" * 45)

        print("SPACE : Pause / Resume")
        print("R     : Restart mission")
        print("ESC   : Exit")

        try:

            while (
                self.running
                and self.episode < self.episodes
            ):

                self.restart = False

                self.run_episode()

                if not self.running:
                    break

                if self.restart:
                    print("Restarting mission...")
                    continue

                self.episode += 1

        finally:

            self.save_results()

            self.renderer.close()
            self.env.close()


# -----------------------------------------
# COMMAND LINE ENTRY POINT
# -----------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Smart Irrigation RL Mission Player"
    )

    parser.add_argument(
        "--algorithm",
        choices=[
            "DQN",
            "REINFORCE",
            "PPO",
            "A2C"
        ],
        default="DQN"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--output",
        type=str,
        default="logs/mission_results.csv"
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("Episodes must be at least 1.")

    if args.speed < 0:
        parser.error("Speed cannot be negative.")

    mission = IrrigationMission(
        algorithm=args.algorithm,
        episodes=args.episodes,
        seed=args.seed,
        speed=args.speed,
        output=args.output
    )

    mission.run()


if __name__ == "__main__":
    main()