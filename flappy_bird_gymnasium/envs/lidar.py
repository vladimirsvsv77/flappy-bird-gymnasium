import numpy as np
import pygame

from flappy_bird_gymnasium.envs.constants import (
    BASE_HEIGHT,
    BASE_WIDTH,
    PIPE_HEIGHT,
    PIPE_WIDTH,
    PLAYER_ROT_THR,
)


class LIDAR:
    def __init__(self, max_distance):
        self._max_distance = max_distance
        self.collisions = np.zeros((180, 2))

    def scan(self, player_x, player_y, player_rot, upper_pipes, lower_pipes, ground):
        result = np.empty([180])

        # sort pipes from nearest to farthest
        upper_pipes = sorted(upper_pipes, key=lambda pipe: pipe["x"])
        lower_pipes = sorted(lower_pipes, key=lambda pipe: pipe["x"])

        # get collisions with precision 1 degree
        for i, angle in enumerate(range(0, 180, 1)):
            # Getting player's rotation
            visible_rot = PLAYER_ROT_THR
            if player_rot <= PLAYER_ROT_THR:
                visible_rot = player_rot

            rad = np.radians(angle - 90 - visible_rot)
            x = self._max_distance * np.cos(rad) + player_x
            y = self._max_distance * np.sin(rad) + player_y
            line = (player_x, player_y, x, y)
            self.collisions[i] = (x, y)

            # check ground collision
            ground_rect = pygame.Rect(0, ground["y"], BASE_WIDTH, BASE_HEIGHT)
            collision = ground_rect.clipline(line)
            # print("gound collision: ", collision, " ", angle, "line: ", line)
            if collision:
                self.collisions[i] = collision[0]

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
                    self.collisions[i] = collision_A[0]
                    break
                elif collision_B:
                    self.collisions[i] = collision_B[0]
                    break

            # calculate distance
            result[i] = np.sqrt(
                (player_x - self.collisions[i][0]) ** 2
                + (player_y - self.collisions[i][1]) ** 2
            )

        return result
