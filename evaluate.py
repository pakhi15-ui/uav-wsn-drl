import numpy as np
import matplotlib.pyplot as plt
from env.wsn_env import UAVWSNEnv
from stable_baselines3 import PPO
import os

os.makedirs("results", exist_ok=True)

env = UAVWSNEnv(n_sensors=10, grid_size=100, max_steps=200)
model = PPO.load("models/trained_uav_agent", env=env)

def run_episode(use_agent=True):
    obs, _ = env.reset()
    total_reward = 0
    uav_path = [(env.uav_x, env.uav_y)]
    for _ in range(200):
        if use_agent:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        uav_path.append((env.uav_x, env.uav_y))
        if done:
            break
    return total_reward, uav_path

print("Running 20 evaluation episodes...")
agent_rewards, random_rewards = [], []
for i in range(20):
    r, _ = run_episode(use_agent=True)
    agent_rewards.append(r)
    r2, _ = run_episode(use_agent=False)
    random_rewards.append(r2)

print(f"\nDRL Agent  — Avg Reward: {np.mean(agent_rewards):.1f} | Best: {max(agent_rewards):.1f}")
print(f"Random     — Avg Reward: {np.mean(random_rewards):.1f} | Best: {max(random_rewards):.1f}")
print(f"Improvement: {((np.mean(agent_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['DRL Agent', 'Random'], [np.mean(agent_rewards), np.mean(random_rewards)],
            color=['steelblue', 'salmon'], width=0.4)
axes[0].set_title('Average Reward: DRL Agent vs Random')
axes[0].set_ylabel('Total Reward')
axes[0].grid(axis='y')

_, agent_path = run_episode(use_agent=True)
sensor_x = [s.x for s in env.sensors]
sensor_y = [s.y for s in env.sensors]
path_x = [p[0] for p in agent_path]
path_y = [p[1] for p in agent_path]
axes[1].scatter(sensor_x, sensor_y, s=100, c='red', zorder=5, label='Sensors')
axes[1].plot(path_x, path_y, 'b-', alpha=0.6, linewidth=1.5, label='UAV Path')
axes[1].plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
axes[1].plot(path_x[-1], path_y[-1], 'r^', markersize=10, label='End')
axes[1].set_title('UAV Flight Path (DRL Agent)')
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 100)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/evaluation.png", dpi=150)
print("\nResults saved to results/evaluation.png")
plt.show()
