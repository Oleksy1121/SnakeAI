import gymnasium as gym
from env.snake_env import SnakeEnv
import numpy as np
import pygame
from matplotlib import pyplot as plt

pygame.init()

env = SnakeEnv(verbose=1)
env.max_apples = 10
env.difficulty = 1
env.initial_length = 1

# Env reset
observation_dict, info = env.reset()

terminated = False
truncated = False

# Visualization
plt.ion()  
fig, (ax_image, ax_vector) = plt.subplots(1, 2, figsize=(10, 5))

image_obs = observation_dict["image"]
im = ax_image.imshow(image_obs.squeeze(), cmap='gray', vmin=0, vmax=255)
ax_image.set_title('Plansza')
ax_image.axis('off')

vector_labels = [
    "scaled_distance", "cos_angle", "sin_angle", "snake_length",
    "dir_UP", "dir_RIGHT", "dir_DOWN", "dir_LEFT",
    "f_wall", "f_apple", "f_tail", "f_dist",
    "l_wall", "l_apple", "l_tail", "l_dist",
    "r_wall", "r_apple", "r_tail", "r_dist"
]

vector_obs = observation_dict["vector"]
ax_vector.bar(np.arange(len(vector_obs)), vector_obs, color='blue')
ax_vector.set_title('Dane wektorowe')
ax_vector.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()

# Game loop
while not terminated and not truncated:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    env.render()
    random_action = env.action_space.sample()
    observation_dict, reward, terminated, truncated, info = env.step(0)

    im.set_data(observation_dict["image"].squeeze())

    ax_vector.clear()
    vector_obs = observation_dict["vector"]
    ax_vector.bar(np.arange(len(vector_obs)), vector_obs, color='blue')
    ax_vector.set_title('Dane wektorowe')
    ax_vector.set_ylim(0, 1.0)
    ax_vector.set_xticks(np.arange(len(vector_obs)))
    ax_vector.set_xticklabels(vector_labels, rotation=45, ha='right')

    fig.canvas.draw()
    fig.canvas.flush_events()
    
    pygame.time.wait(100)
