from typing import cast

import gymnasium as gym
import numpy as np
import pygame

from environment.car_model import KinematicBicycleModel

# Register and create environment
gym.register(
    id="LTS-v0",
    entry_point="environment.lts_env:LtsEnv",
)

car_model = KinematicBicycleModel()

env = gym.make(
    "LTS-v0",
    render_mode="human",
    track_csv_path="data/tracks/MexicoCity.csv",
    num_rangefinders=20,
    num_future_curvatures=20,
    car_model=car_model,
    follow_car=True,
)


# Reset environment and initialize pygame rendering to receive keyboard input
observation, info = env.reset()
env.render()

# Control senstivity
THROTTLE_SENSITIVITY = 0.15
STEERING_SENSITIVITY = 0.03

# Decay factors for when keys are released
THROTTLE_DECAY = 0.95
STEERING_DECAY = 0.95

action_space = cast(gym.spaces.Box, env.action_space)
action = np.zeros(action_space.shape)

running = True
while running:
    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:  # Brake
        action[0] -= THROTTLE_SENSITIVITY
    elif keys[pygame.K_UP]:  # Accelerate. Can't do both at the same time.
        action[0] += THROTTLE_SENSITIVITY
    if keys[pygame.K_RIGHT]:  # Steer right
        action[1] -= STEERING_SENSITIVITY
    if keys[pygame.K_LEFT]:  # Steer left
        action[1] += STEERING_SENSITIVITY

    # Decay throttle and steering when keys are released
    action[0] *= THROTTLE_DECAY
    action[1] *= STEERING_DECAY

    # Clip action to valid range
    action = np.clip(action, action_space.low, action_space.high)

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()
