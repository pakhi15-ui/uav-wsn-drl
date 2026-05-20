import numpy as np
import matplotlib.pyplot as plt
from env.wsn_env import UAVWSNEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current = []
    def _on_step(self):
        self.current.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self.current))
            self.current = []
        return True

env = UAVWSNEnv(n_sensors=10, grid_size=100, max_steps=200)
model = PPO("MlpPolicy", env,
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=128,
            n_epochs=15,
            gamma=0.995,
            verbose=1)

cb = RewardCallback()
print("Training 200,000 steps...")
model.learn(total_timesteps=200000, callback=cb, progress_bar=True)
model.save("models/trained_uav_agent")
print("Saved!")

rewards = cb.episode_rewards
if len(rewards) > 10:
    plt.figure(figsize=(10,4))
    plt.plot(rewards, alpha=0.3, color='blue')
    w = 15
    smoothed = np.convolve(rewards, np.ones(w)/w, mode='valid')
    plt.plot(range(w-1, len(rewards)), smoothed, color='red', lw=2, label='Smoothed')
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("Training Progress"); plt.grid(True); plt.legend()
    plt.savefig("results/training_final.png", dpi=150)
    plt.show()
    print(f"Final avg reward (last 20 eps): {np.mean(rewards[-20:]):.1f}")
