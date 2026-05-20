import numpy as np
import matplotlib.pyplot as plt
from env.wsn_env import UAVWSNEnv
from models.drl_agent import UAVAgent
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 50)
print("UAV WSN - LSTM + Deep Reinforcement Learning")
print("=" * 50)

# Create environment
env = UAVWSNEnv(n_sensors=10, grid_size=100, max_steps=200)
print(f"Environment ready | Sensors: 10 | Grid: 100x100")

# Create agent
agent = UAVAgent(env=env, n_sensors=10)
print("Agent ready | PPO + LSTM initialized")
print()

# Train
callback = agent.train(total_timesteps=50000)

# Save the trained agent
agent.save("models/trained_uav_agent")

# Plot training rewards
rewards = callback.episode_rewards
if len(rewards) > 0:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.4, color='blue', label='Episode reward')
    # Smooth line
    if len(rewards) >= 10:
        window = 10
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed,
                color='red', linewidth=2, label='Smoothed (10 ep)')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    lengths = callback.episode_lengths
    plt.plot(lengths, alpha=0.4, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/training_progress.png", dpi=150)
    print("Training graph saved to results/training_progress.png")
    plt.show()

print()
print("Month 2 Complete!")
print("Trained agent saved. Ready for evaluation in Month 3.")
