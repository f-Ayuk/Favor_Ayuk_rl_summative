import pygame
import numpy as np


class FarmRenderer:

    def __init__(self):
        pygame.init()

        self.width = 950
        self.height = 620

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

        pygame.display.set_caption(
            "Smart Irrigation RL Simulator"
        )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.small_font = pygame.font.SysFont("Arial", 17)

    def draw(self, env):

        self.screen.fill((235, 242, 225))

        title = self.font.render(
            "SMART IRRIGATION AGRICULTURAL SIMULATOR",
            True, (25, 65, 35)
        )

        self.screen.blit(title, (150, 20))

        for i in range(4):

            x = 65 + (i % 2) * 430
            y = 100 + (i // 2) * 210

            moisture = float(env.moisture[i])
            health = float(env.health[i])

            # Moisture determines soil colour.
            color = (
                int(125 - moisture * 65),
                int(85 + moisture * 75),
                int(45 + moisture * 45)
            )

            pygame.draw.rect(
                self.screen,
                color,
                (x, y, 370, 150),
                border_radius=12
            )

            # Crop plants.
            for plant in range(8):

                px = x + 35 + plant * 42
                py = y + 85

                crop_color = (
                    35,
                    int(90 + health * 120),
                    35
                )

                pygame.draw.line(
                    self.screen,
                    crop_color,
                    (px, py),
                    (px, py - 30 * health),
                    5
                )

                pygame.draw.circle(
                    self.screen,
                    crop_color,
                    (px, int(py - 30 * health)),
                    7
                )

            label = self.font.render(
                f"Zone {i+1}",
                True, (255, 255, 255)
            )

            self.screen.blit(label, (x + 10, y + 8))

            text = self.small_font.render(
                f"Moisture: {moisture:.2f} | Health: {health:.2f}",
                True, (255, 255, 255)
            )

            self.screen.blit(text, (x + 10, y + 120))

        pygame.draw.rect(
            self.screen, (40, 90, 180),
            (65, 535, 800 * env.reservoir, 22)
        )

        pygame.draw.rect(
            self.screen, (30, 30, 30),
            (65, 535, 800, 22), 2
        )

        status = self.font.render(
            f"Day: {env.day}/{env.max_days}   "
            f"Reservoir: {env.reservoir:.2%}",
            True, (25, 65, 35)
        )

        self.screen.blit(status, (65, 570))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()