import numpy as np
import matplotlib.pyplot as plt
from env.wsn_env import UAVWSNEnv
from models.drl_agent import UAVAgent
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

env = UAVWSNEnv(n_sensors=10, grid_size=100, max_steps=200)
agent = UAVAgent(env=env, n_sensors=10)

print("Training for 200,000 steps...")
callback = agent.train(total_timesteps=200000)
agent.save("models/trained_uav_agent")

rewards = callback.episode_rewards
if len(rewards) > 0:
    plt.figure(figsize=(10, 4))
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, color='red', linewidth=2)
    plt.plot(rewards, alpha=0.3, color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress (200k steps)")
    plt.grid(True)
    plt.savefig("results/training_progress2.png", dpi=150)
    plt.show()

print("Done! Now run: python3 evaluate.py")
