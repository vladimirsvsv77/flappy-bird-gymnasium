import numpy as np
import pygame

from flappy_bird_gymnasium.envs.constants import (
    BASE_HEIGHT,
    BASE_WIDTH,
    PIPE_HEIGHT,
    PIPE_WIDTH,
    PLAYER_ROT_THR,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
)


class LIDAR:
    def __init__(self, max_distance, rotation):
        self._max_distance = max_distance
        self._rotation = rotation
        self.collisions = np.zeros((180, 2))

    def draw(self, surface, player_x, player_y):
        for i in range(self.collisions.shape[0]):
            if self._rotation == 0:
                pygame.draw.line(
                    surface,
                    "red",
                    (
                        player_x + PLAYER_WIDTH,
                        player_y + (PLAYER_HEIGHT / 2),
                    ),
                    (
                        self.collisions[i][0],
                        self.collisions[i][1],
                    ),
                    1,
                )
            elif self._rotation == 180:
                pygame.draw.line(
                    surface,
                    "red",
                    (
                        player_x,
                        player_y + (PLAYER_HEIGHT / 2),
                    ),
                    (
                        self.collisions[i][0],
                        self.collisions[i][1],
                    ),
                    1,
                )
            else:
                raise ValueError()
                
    def scan(self, player_x, player_y, player_rot, upper_pipes, lower_pipes, ground):
        result = np.empty([180])

        # sort pipes from nearest to farthest
        upper_pipes = sorted(upper_pipes, key=lambda pipe: pipe["x"])
        lower_pipes = sorted(lower_pipes, key=lambda pipe: pipe["x"])

        if self._rotation == 0:
            offset_x = player_x + PLAYER_WIDTH
            offset_y = player_y + (PLAYER_HEIGHT / 2)
        elif self._rotation == 180:
            offset_x = player_x
            offset_y = player_y + (PLAYER_HEIGHT / 2)
        else:
            raise ValueError()

        # Getting player's rotation
        visible_rot = PLAYER_ROT_THR
        if player_rot <= PLAYER_ROT_THR:
            visible_rot = player_rot

        # get collisions with precision 1 degree
        for i, angle in enumerate(range(0, 180, 1)):
            rad = np.radians(angle - 90 - visible_rot - self._rotation)
            x = self._max_distance * np.cos(rad) + offset_x
            y = self._max_distance * np.sin(rad) + offset_y
            line = (offset_x, offset_y, x, y)
            self.collisions[i] = (x, y) if y < ground["y"] else (x, ground["y"])

            # check ground collision
            ground_rect = pygame.Rect(0, ground["y"], BASE_WIDTH, BASE_HEIGHT)
            collision = ground_rect.clipline(line)
            if collision:
                self.collisions[i] = collision[0] if collision[0][1] < ground["y"] else (
                    collision[0][0],
                    ground["y"],
                )

            # check pipe collision
            for up_pipe, low_pipe in zip(upper_pipes, lower_pipes):
                # upper and lower pipe rects
                up_pipe_rect = pygame.Rect(
                    up_pipe["x"], up_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )
                low_pipe_rect = pygame.Rect(
                    low_pipe["x"], low_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )

                # check collision
                collision_A = up_pipe_rect.clipline(line)
                collision_B = low_pipe_rect.clipline(line)

                if collision_A:
                    collision_A = collision_A[0]
                    collision_A = collision_A if collision_A[0] <= up_pipe["x"] else (
                        up_pipe["x"],
                        collision_A[1],
                    )
                    collision_A = collision_A if player_x < up_pipe["x"] or player_x > up_pipe["x"] else (
                        collision_A[0],
                        up_pipe["y"],
                    )
                    self.collisions[i] = collision_A
                    break
                elif collision_B:
                    collision_B = collision_B[0]
                    collision_B = collision_B if collision_B[0] <= low_pipe["x"] else (
                        low_pipe["x"],
                        collision_B[1],
                    )
                    collision_B = collision_B if collision_B[1] <= ground["y"] else (
                        collision_B[0],
                        ground["y"],
                    )
                    collision_B = collision_B if player_x < low_pipe["x"] or player_x > low_pipe["x"] else (
                        collision_B[0],
                        low_pipe["y"],
                    )
                    self.collisions[i] = collision_B
                    break

            # calculate distance
            result[i] = np.sqrt(
                (offset_x - self.collisions[i][0]) ** 2
                + (offset_y - self.collisions[i][1]) ** 2
            )

        return result
