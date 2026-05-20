import torch
import torch.nn as nn
import numpy as np

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class SensorDataBuffer:
    """Stores recent sensor history for LSTM input"""
    def __init__(self, n_sensors=10, history_len=10):
        self.n_sensors = n_sensors
        self.history_len = history_len
        self.history = [[] for _ in range(n_sensors)]

    def update(self, sensor_states):
        """Add latest sensor readings to history"""
        for i, state in enumerate(sensor_states):
            self.history[i].append(state)
            if len(self.history[i]) > self.history_len:
                self.history[i].pop(0)

    def get_tensor(self, sensor_id):
        """Get history as tensor for LSTM input"""
        hist = self.history[sensor_id]
        if len(hist) < self.history_len:
            pad = [hist[0]] * (self.history_len - len(hist))
            hist = pad + hist
        return torch.FloatTensor(hist).unsqueeze(0)

    def predict_demand(self, model, sensor_id):
        """Use LSTM to predict next data demand for a sensor"""
        tensor = self.get_tensor(sensor_id)
        with torch.no_grad():
            prediction = model(tensor)
        return prediction.item()
