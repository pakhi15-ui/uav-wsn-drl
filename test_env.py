from env.wsn_env import UAVWSNEnv
import numpy as np

env = UAVWSNEnv(n_sensors=10, grid_size=100)
obs, _ = env.reset()

print(f"Environment created!")
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print()

total_reward = 0
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    print(f"Step {i+1} | Reward: {reward:.2f} | Done: {done}")

print()
print(f"Total reward (random agent): {total_reward:.2f}")
print("Environment is working correctly!")
