import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.sensor_node import SensorNode

class UAVWSNEnv(gym.Env):
    def __init__(self, n_sensors=10, grid_size=100, max_steps=200):
        super().__init__()
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.uav_x = grid_size / 2
        self.uav_y = grid_size / 2
        self.uav_energy = 200.0
        self.visited_sensors = set()

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        obs_size = 3 + (n_sensors * 3)
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

    def _get_obs(self):
        uav_state = [
            self.uav_x / self.grid_size,
            self.uav_y / self.grid_size,
            self.uav_energy / 200.0
        ]
        sensor_states = []
        for s in self.sensors:
            sensor_states.append((s.x - self.uav_x) / self.grid_size)
            sensor_states.append((s.y - self.uav_y) / self.grid_size)
            sensor_states.append(min(s.data_buffer / 10.0, 1.0))
        return np.array(uav_state + sensor_states, dtype=np.float32)

    def step(self, action):
        dx = float(action[0]) * 8.0
        dy = float(action[1]) * 8.0
        self.uav_x = np.clip(self.uav_x + dx, 0, self.grid_size)
        self.uav_y = np.clip(self.uav_y + dy, 0, self.grid_size)
        self.uav_energy -= 0.5

        for sensor in self.sensors:
            sensor.generate_data(amount=1.0)

        step_collected = 0
        for i, sensor in enumerate(self.sensors):
            dist = np.sqrt((self.uav_x - sensor.x)**2 +
                          (self.uav_y - sensor.y)**2)
            if dist < 20:
                collected = sensor.transmit_data(dist, 10.0)
                step_collected += collected
                # Bonus for visiting a new sensor
                if i not in self.visited_sensors:
                    self.visited_sensors.add(i)
                    step_collected += 5.0

        reward = step_collected * 3.0

        self.current_step += 1
        done = (self.current_step >= self.max_steps or self.uav_energy <= 0)

        # Big bonus at end for coverage
        if done:
            coverage = len(self.visited_sensors) / self.n_sensors
            reward += coverage * 200.0

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav_x = self.grid_size / 2
        self.uav_y = self.grid_size / 2
        self.uav_energy = 200.0
        self.current_step = 0
        self.visited_sensors = set()
        self._init_sensors()
        return self._get_obs(), {}

    def render(self):
        pass
