import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SmartIrrigationEnv(gym.Env):
    """
    Four-zone smart irrigation environment.

    One action irrigates one zone, or turns irrigation off.
    Each episode simulates a 60-day growing season.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_days=60, seed=None):
        super().__init__()

        self.n_zones = 4
        self.max_days = max_days
        self.render_mode = render_mode

        # 0 = OFF
        # 1-3 = Zone 1, low/medium/high
        # 4-6 = Zone 2, low/medium/high
        # 7-9 = Zone 3, low/medium/high
        # 10-12 = Zone 4, low/medium/high
        self.action_space = spaces.Discrete(13)

        # 4 moisture + 4 crop health + 7 global features
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)

        self.water_levels = np.array([0, 5, 10, 15])
        self.water_cost = 0.0008

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.day = 0

        self.moisture = np.full(4, 0.55, dtype=np.float32)
        self.health = np.full(4, 0.90, dtype=np.float32)

        self.reservoir = 1.0
        self.temperature = 0.50
        self.rainfall = 0.0
        self.rain_forecast = 0.25
        self.energy_price = 0.5

        self.total_water = 0.0
        self.total_energy = 0.0
        self.total_rainfall = 0.0

        return self._get_obs(), {
            "day": self.day,
            "water_used": self.total_water
        }

    def _get_obs(self):
        growth = self.day / self.max_days

        obs = np.concatenate([
            self.moisture,
            self.health,
            np.array([
                self.reservoir,
                self.rainfall,
                self.temperature,
                self.rain_forecast,
                growth,
                self.day / self.max_days,
                self.energy_price
            ])
        ])

        return np.clip(obs, 0, 1).astype(np.float32)

    def step(self, action):

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError("Invalid irrigation action")

        water_applied = 0.0
        energy_used = 0.0
        irrigated_zone = None

        # Decode irrigation action.
        if action != 0:

            irrigated_zone = (action - 1) // 3
            irrigation_level = (action - 1) % 3 + 1

            requested_water = self.water_levels[irrigation_level]

            # Water is limited by reservoir capacity.
            available_water = self.reservoir * 4000

            water_applied = min(
                requested_water,
                available_water
            )

            self.reservoir -= water_applied / 4000

            # Convert irrigation into soil moisture.
            self.moisture[irrigated_zone] += (
                water_applied / 100
            )

            energy_used = water_applied * (
                0.04 + self.energy_price * 0.06
            )

        self.moisture = np.clip(
            self.moisture, 0, 1
        )

        self.total_water += water_applied
        self.total_energy += energy_used

        # Stochastic environmental conditions.
        self.temperature = np.clip(
            self.rng.normal(0.55, 0.13), 0, 1
        )

        self.rainfall = self.rng.choice(
            [0.0, 0.05, 0.12, 0.20],
            p=[0.55, 0.20, 0.17, 0.08]
        )

        self.rain_forecast = self.rng.uniform(0, 1)

        self.energy_price = self.rng.uniform(0.2, 1.0)

        self.total_rainfall += self.rainfall

        # Rainfall increases soil moisture in all zones.
        self.moisture += self.rainfall

        # Evapotranspiration increases with temperature.
        evaporation = 0.025 + self.temperature * 0.035

        self.moisture -= evaporation

        self.moisture = np.clip(
            self.moisture, 0, 1
        )

        # Crop water stress.
        stress = np.maximum(
            0, 0.45 - self.moisture
        )

        # Excess moisture causes waterlogging.
        excess = np.maximum(
            0, self.moisture - 0.80
        )

        # Crop health evolves over time.
        health_change = (
            0.015
            - 0.12 * stress
            - 0.10 * excess
        )

        self.health += health_change

        self.health = np.clip(
            self.health, 0, 1
        )

        # Crop growth stage.
        growth = self.day / self.max_days

        # Reward components.
        average_health = np.mean(self.health)
        average_stress = np.mean(stress)
        average_excess = np.mean(excess)

        water_penalty = water_applied / 15
        energy_penalty = energy_used / 2

        reward = (
            5.0 * average_health
            - 4.0 * average_stress
            - 3.0 * average_excess
            - 0.5 * water_penalty
            - 0.3 * energy_penalty
        )

        # Small terminal bonus for a successful harvest.
        self.day += 1

        terminated = (
            np.mean(self.health) < 0.10
            or self.reservoir <= 0.001
        )

        truncated = self.day >= self.max_days

        if terminated or truncated:
            reward += 10.0 * np.mean(self.health)

        info = {
            "day": self.day,
            "average_health": float(average_health),
            "average_moisture": float(np.mean(self.moisture)),
            "water_used": float(self.total_water),
            "energy_used": float(self.total_energy),
            "reservoir": float(self.reservoir),
            "rainfall": float(self.rainfall),
            "irrigated_zone": irrigated_zone,
            "water_applied": float(water_applied),
            "stress": float(average_stress)
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info
        )