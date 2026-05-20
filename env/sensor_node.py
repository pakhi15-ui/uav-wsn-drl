import numpy as np

class SensorNode:
    def __init__(self, node_id, x, y, initial_energy=100.0):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.energy = initial_energy
        self.max_energy = initial_energy
        self.data_buffer = 0.0
        self.is_active = True

    def generate_data(self, amount=None):
        if amount is None:
            amount = np.random.uniform(0.5, 2.0)
        self.data_buffer += amount

    def transmit_data(self, uav_distance, bandwidth_allocated):
        energy_cost = 0.01 * uav_distance * bandwidth_allocated
        if self.energy >= energy_cost and self.data_buffer > 0:
            collected = min(self.data_buffer, bandwidth_allocated)
            self.data_buffer -= collected
            self.energy -= energy_cost
            return collected
        return 0.0

    def get_state(self):
        return [
            self.x, self.y,
            self.energy / self.max_energy,
            self.data_buffer,
            float(self.is_active)
        ]
