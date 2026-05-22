import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.sensor_node import SensorNode
from env.obstacles import ObstacleManager

class MultiUAVEnv(gym.Env):
    """Two UAVs learn to divide the sensor field and coordinate"""
    def __init__(self, n_sensors=10, grid_size=100, max_steps=200, n_uavs=2):
        super().__init__()
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_uavs = n_uavs
        self.current_step = 0
        self.visited_sensors = set()
        self.dead_sensors = set()

        # UAVs start at different corners
        self.uav_positions = [
            [grid_size * 0.25, grid_size * 0.5],
            [grid_size * 0.75, grid_size * 0.5],
        ]
        self.uav_energies = [200.0] * n_uavs

        # Combined action for all UAVs: 2 values per UAV
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(n_uavs * 2,), dtype=np.float32
        )

        # obs: all UAV states + sensor states
        obs_size = (n_uavs * 3) + (n_sensors * 5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )
        self._init_sensors()

    def _init_sensors(self):
        self.sensors = [
            SensorNode(
                node_id=i,
                x=np.random.uniform(10, self.grid_size - 10),
                y=np.random.uniform(10, self.grid_size - 10),
                initial_energy=100.0
            )
            for i in range(self.n_sensors)
        ]

    def _get_urgency(self, sensor):
        energy_u = 1.0 - (sensor.energy / sensor.max_energy)
        buffer_u = min(sensor.data_buffer / 20.0, 1.0)
        return energy_u * 0.6 + buffer_u * 0.4

    def _get_obs(self):
        uav_states = []
        for i in range(self.n_uavs):
            uav_states += [
                self.uav_positions[i][0] / self.grid_size,
                self.uav_positions[i][1] / self.grid_size,
                self.uav_energies[i] / 200.0
            ]
        sensor_states = []
        for s in self.sensors:
            # Relative to UAV 0
            sensor_states.append((s.x - self.uav_positions[0][0]) / self.grid_size)
            sensor_states.append((s.y - self.uav_positions[0][1]) / self.grid_size)
            sensor_states.append(min(s.data_buffer / 10.0, 1.0))
            sensor_states.append(s.energy / s.max_energy)
            sensor_states.append(self._get_urgency(s))
        return np.array(uav_states + sensor_states, dtype=np.float32)

    def step(self, action):
        for u in range(self.n_uavs):
            dx = float(action[u*2]) * 8.0
            dy = float(action[u*2+1]) * 8.0
            self.uav_positions[u][0] = np.clip(
                self.uav_positions[u][0] + dx, 0, self.grid_size)
            self.uav_positions[u][1] = np.clip(
                self.uav_positions[u][1] + dy, 0, self.grid_size)
            self.uav_energies[u] -= 0.5

        for sensor in self.sensors:
            if sensor.node_id not in self.dead_sensors:
                sensor.generate_data(amount=1.0)
                sensor.energy -= 0.1
                if sensor.energy <= 0:
                    sensor.energy = 0
                    sensor.is_active = False
                    self.dead_sensors.add(sensor.node_id)

        total_collected = 0
        for i, sensor in enumerate(self.sensors):
            if i in self.dead_sensors:
                continue
            for u in range(self.n_uavs):
                dist = np.sqrt(
                    (self.uav_positions[u][0] - sensor.x)**2 +
                    (self.uav_positions[u][1] - sensor.y)**2
                )
                if dist < 20:
                    urgency = self._get_urgency(sensor)
                    collected = sensor.transmit_data(dist, 10.0)
                    total_collected += collected * (1.0 + urgency)
                    if i not in self.visited_sensors:
                        self.visited_sensors.add(i)
                        total_collected += 5.0
                    break  # only one UAV collects per sensor per step

        # Penalize overlap between UAVs
        uav_dist = np.sqrt(
            (self.uav_positions[0][0] - self.uav_positions[1][0])**2 +
            (self.uav_positions[0][1] - self.uav_positions[1][1])**2
        )
        overlap_penalty = max(0, 20 - uav_dist)

        reward = total_collected * 3.0
        reward -= len(self.dead_sensors) * 2.0
        reward -= overlap_penalty * 0.5

        self.current_step += 1
        done = (self.current_step >= self.max_steps or
                all(e <= 0 for e in self.uav_energies))

        if done:
            coverage = len(self.visited_sensors) / self.n_sensors
            survival = 1.0 - (len(self.dead_sensors) / self.n_sensors)
            reward += coverage * 200.0 + survival * 100.0

        return self._get_obs(), reward, done, False, {
            "visited": len(self.visited_sensors),
            "dead_sensors": len(self.dead_sensors),
            "uav_overlap_penalty": overlap_penalty
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav_positions = [
            [self.grid_size * 0.25, self.grid_size * 0.5],
            [self.grid_size * 0.75, self.grid_size * 0.5],
        ]
        self.uav_energies = [200.0] * self.n_uavs
        self.current_step = 0
        self.visited_sensors = set()
        self.dead_sensors = set()
        self._init_sensors()
        return self._get_obs(), {}

    def render(self):
        pass
