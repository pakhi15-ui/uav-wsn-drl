import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.sensor_node import SensorNode
from env.obstacles import ObstacleManager

class UAVWSNEnvV2(gym.Env):
    def __init__(self, n_sensors=10, grid_size=100, max_steps=200, n_obstacles=4):
        super().__init__()
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_obstacles = n_obstacles
        self.current_step = 0
        self.uav_x = grid_size / 2
        self.uav_y = grid_size / 2
        self.uav_energy = 200.0
        self.visited_sensors = set()
        self.dead_sensors = set()
        self.collisions = 0

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        # UAV(3) + sensors(5 each) + obstacles(2 each)
        obs_size = 3 + (n_sensors * 5) + (n_obstacles * 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )
        self._init_sensors()
        self.obstacle_mgr = ObstacleManager(n_obstacles, grid_size)

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
        energy_urgency = 1.0 - (sensor.energy / sensor.max_energy)
        buffer_urgency = min(sensor.data_buffer / 20.0, 1.0)
        return energy_urgency * 0.6 + buffer_urgency * 0.4

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
            sensor_states.append(s.energy / s.max_energy)
            sensor_states.append(self._get_urgency(s))
        obstacle_states = self.obstacle_mgr.get_states()
        return np.array(uav_state + sensor_states + obstacle_states, dtype=np.float32)

    def step(self, action):
        self.obstacle_mgr.step()

        dx = float(action[0]) * 8.0
        dy = float(action[1]) * 8.0
        new_x = np.clip(self.uav_x + dx, 0, self.grid_size)
        new_y = np.clip(self.uav_y + dy, 0, self.grid_size)

        collision = self.obstacle_mgr.check_collision(new_x, new_y)
        if not collision:
            self.uav_x = new_x
            self.uav_y = new_y
        else:
            self.collisions += 1

        self.uav_energy -= 0.5

        for sensor in self.sensors:
            if sensor.node_id not in self.dead_sensors:
                sensor.generate_data(amount=1.0)
                sensor.energy -= 0.1
                if sensor.energy <= 0:
                    sensor.energy = 0
                    sensor.is_active = False
                    self.dead_sensors.add(sensor.node_id)

        step_collected = 0
        for i, sensor in enumerate(self.sensors):
            if i in self.dead_sensors:
                continue
            dist = np.sqrt((self.uav_x - sensor.x)**2 +
                          (self.uav_y - sensor.y)**2)
            if dist < 20:
                urgency = self._get_urgency(sensor)
                collected = sensor.transmit_data(dist, 10.0)
                step_collected += collected * (1.0 + urgency * 2.0)
                if i not in self.visited_sensors:
                    self.visited_sensors.add(i)
                    step_collected += 5.0

        reward = step_collected * 3.0
        reward -= len(self.dead_sensors) * 2.0
        reward -= collision * 5.0

        self.current_step += 1
        done = (self.current_step >= self.max_steps or self.uav_energy <= 0)

        if done:
            coverage = len(self.visited_sensors) / self.n_sensors
            survival = 1.0 - (len(self.dead_sensors) / self.n_sensors)
            reward += coverage * 200.0 + survival * 100.0

        return self._get_obs(), reward, done, False, {
            "dead_sensors": len(self.dead_sensors),
            "visited": len(self.visited_sensors),
            "collisions": self.collisions
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav_x = self.grid_size / 2
        self.uav_y = self.grid_size / 2
        self.uav_energy = 200.0
        self.current_step = 0
        self.visited_sensors = set()
        self.dead_sensors = set()
        self.collisions = 0
        self._init_sensors()
        self.obstacle_mgr.reset()
        return self._get_obs(), {}

    def render(self):
        pass
