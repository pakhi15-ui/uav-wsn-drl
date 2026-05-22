import numpy as np

class Obstacle:
    def __init__(self, x, y, radius=8, vx=None, vy=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = vx if vx else np.random.uniform(-1, 1)
        self.vy = vy if vy else np.random.uniform(-1, 1)

    def move(self, grid_size=100):
        self.x += self.vx
        self.y += self.vy
        if self.x <= 0 or self.x >= grid_size:
            self.vx *= -1
        if self.y <= 0 or self.y >= grid_size:
            self.vy *= -1
        self.x = np.clip(self.x, 0, grid_size)
        self.y = np.clip(self.y, 0, grid_size)

    def contains(self, x, y):
        return np.sqrt((x - self.x)**2 + (y - self.y)**2) < self.radius

class ObstacleManager:
    def __init__(self, n_obstacles=4, grid_size=100):
        self.grid_size = grid_size
        self.obstacles = [
            Obstacle(
                x=np.random.uniform(20, 80),
                y=np.random.uniform(20, 80)
            )
            for _ in range(n_obstacles)
        ]

    def step(self):
        for obs in self.obstacles:
            obs.move(self.grid_size)

    def check_collision(self, x, y):
        return any(obs.contains(x, y) for obs in self.obstacles)

    def get_states(self):
        states = []
        for obs in self.obstacles:
            states.extend([
                obs.x / self.grid_size,
                obs.y / self.grid_size
            ])
        return states

    def reset(self):
        self.obstacles = [
            Obstacle(
                x=np.random.uniform(20, 80),
                y=np.random.uniform(20, 80)
            )
            for _ in range(len(self.obstacles))
        ]
