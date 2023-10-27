# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Tests the simple-observations version of the Flappy Bird environment with a
human player.
"""

import gymnasium
import pygame

import flappy_bird_gymnasium
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


def play():
    env = gymnasium.make("FlappyBird-v0", audio_on=True, render_mode="human")

    score, steps = 0, 0
    video_buffer = []

    obs = env.reset()
    while True:
        # Getting action:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and (
                event.key == pygame.K_SPACE or event.key == pygame.K_UP
            ):
                action = 1

        # Processing:
        obs, reward, done, _, info = env.step(action)
        video_buffer.append(obs)

        score += reward
        steps += 1
        print(
            f"Obs: {obs}\n" f"Action: {action}\n" f"Score: {score}\n Steps: {steps}\n"
        )

        if done:
            break

    # Saving video
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection="polar")
    ax2 = fig.add_subplot(122)

    x = np.linspace((np.pi / 2), -(np.pi / 2), 180)
    y = np.array(video_buffer)[:, :180]
    (line,) = ax.plot(x, y[0], "-")  # ax.scatter(x, y[0, :180])
    ax.set_ylim([0, 1])
    ax.set_title("LIDAR scan", fontdict={"fontweight": "bold"})

    x2 = np.arange(steps)
    y2 = np.array(video_buffer)[:, -2]
    y3 = np.array(video_buffer)[:, -1]
    (line2,) = ax2.plot(x2[0], y2[0], "-")
    (line3,) = ax2.plot(x2[0], y3[0], "-")
    ax2.set_xlim([0, steps])
    ax2.set_ylim([-1, 1])
    ax2.set_title("Relief", fontdict={"fontweight": "bold"})

    # save to log file
    df = pd.DataFrame({
        "Time": x2,
        "Position": y2,
        "Velocity": y3,
    })
    df.to_csv("log.csv", index=False)

    def animate(i):
        # RADAR
        line.set_ydata(y[i])

        # RELIEF
        line2.set_xdata(x2[:i])
        line2.set_ydata(y2[:i])
        line3.set_xdata(x2[:i])
        line3.set_ydata(y3[:i])
        return (line, line2, line3)

    anim = animation.FuncAnimation(
        fig, animate, repeat=True, frames=steps, interval=150
    )
    plt.show()

    env.close()


if __name__ == "__main__":
    play()
