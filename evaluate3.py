import numpy as np
import matplotlib.pyplot as plt
from env.wsn_env import UAVWSNEnv
from stable_baselines3 import PPO
import os

os.makedirs("results", exist_ok=True)
env = UAVWSNEnv(n_sensors=10, grid_size=100, max_steps=200)
model = PPO.load("models/trained_uav_agent", env=env)

def run_episode(policy="agent"):
    obs, _ = env.reset()
    total_reward = 0
    uav_path = [(env.uav_x, env.uav_y)]
    for _ in range(200):
        if policy == "agent":
            action, _ = model.predict(obs, deterministic=True)
        elif policy == "random":
            action = env.action_space.sample()
        else:  # greedy
            nearest = min(env.sensors, key=lambda s: (s.x-env.uav_x)**2 + (s.y-env.uav_y)**2)
            dx = np.clip(nearest.x - env.uav_x, -5, 5)
            dy = np.clip(nearest.y - env.uav_y, -5, 5)
            action = np.array([dx, dy], dtype=np.float32)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        uav_path.append((env.uav_x, env.uav_y))
        if done:
            break
    return total_reward, uav_path

print("Running 20 episodes per policy...")
agent_r, random_r, greedy_r = [], [], []
for i in range(20):
    r, _ = run_episode("agent");  agent_r.append(r)
    r, _ = run_episode("random"); random_r.append(r)
    r, _ = run_episode("greedy"); greedy_r.append(r)

print(f"\nDRL Agent — Avg: {np.mean(agent_r):.1f} | Best: {max(agent_r):.1f}")
print(f"Random    — Avg: {np.mean(random_r):.1f} | Best: {max(random_r):.1f}")
print(f"Greedy    — Avg: {np.mean(greedy_r):.1f} | Best: {max(greedy_r):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = ['DRL Agent', 'Random', 'Greedy']
means  = [np.mean(agent_r), np.mean(random_r), np.mean(greedy_r)]
colors = ['steelblue', 'salmon', 'orange']
axes[0].bar(labels, means, color=colors, width=0.4)
axes[0].set_title('Average Reward Comparison')
axes[0].set_ylabel('Total Reward per Episode')
axes[0].grid(axis='y')
for i, v in enumerate(means):
    axes[0].text(i, v + 5, f'{v:.0f}', ha='center', fontweight='bold')

_, path = run_episode("agent")
sensor_x = [s.x for s in env.sensors]
sensor_y = [s.y for s in env.sensors]
axes[1].scatter(sensor_x, sensor_y, s=120, c='red', zorder=5, label='Sensors')
axes[1].plot([p[0] for p in path], [p[1] for p in path], 'b-', alpha=0.7, lw=1.5, label='UAV Path')
axes[1].plot(path[0][0], path[0][1], 'go', markersize=12, label='Start')
axes[1].plot(path[-1][0], path[-1][1], 'r^', markersize=12, label='End')
axes[1].set_title('UAV Flight Path — DRL Agent')
axes[1].set_xlim(0, 100); axes[1].set_ylim(0, 100)
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/evaluation_final.png", dpi=150)
print("Saved to results/evaluation_final.png")
plt.show()
