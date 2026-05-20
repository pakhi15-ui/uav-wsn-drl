import numpy as np
import matplotlib.pyplot as plt
from env.wsn_env import UAVWSNEnv
from stable_baselines3 import PPO
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

env = UAVWSNEnv(n_sensors=10, grid_size=100, max_steps=200)

# Load existing trained model and continue training
model = PPO.load("models/trained_uav_agent", env=env)
print("Loaded existing model, continuing training...")
print("Training for additional 300,000 steps...")

model.learn(total_timesteps=300000, reset_num_timesteps=False, progress_bar=True)
model.save("models/trained_uav_agent")
print("Training complete! Now run: python3 evaluate.py")
