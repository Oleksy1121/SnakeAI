from env.snake_env import SnakeEnv
from stable_baselines3 import PPO
import pygame

pygame.init()

env = SnakeEnv()
env.difficulty = 10

model_path = "models\\PPO-11\\best_model\\best_model.zip"
model = PPO.load(model_path)


for e in range(5):
    observation, info = env.reset() 
    terminated = False
    truncated = False

    while not terminated and not truncated: 
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
 
        
env.close()