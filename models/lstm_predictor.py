import torch
import torch.nn as nn
import numpy as np

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class SensorHistoryTracker:
    """Tracks sensor state history and predicts future demand using LSTM"""
    def __init__(self, n_sensors=10, history_len=10):
        self.n_sensors = n_sensors
        self.history_len = history_len
        self.history = [[] for _ in range(n_sensors)]
        self.model = LSTMPredictor(input_size=5)
        self.predictions = np.zeros(n_sensors)

    def update(self, obs, env):
        """Update history from latest observation"""
        for i, sensor in enumerate(env.sensors):
            state = sensor.get_state()
            self.history[i].append(state)
            if len(self.history[i]) > self.history_len:
                self.history[i].pop(0)

    def predict_all(self, env):
        """Predict urgency score for each sensor"""
        predictions = []
        for i in range(self.n_sensors):
            hist = self.history[i]
            if len(hist) < 3:
                predictions.append(0.5)
                continue
            padded = hist if len(hist) >= self.history_len else \
                     [hist[0]] * (self.history_len - len(hist)) + hist
            tensor = torch.FloatTensor(padded).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(tensor).item()
            predictions.append(max(0, min(1, pred)))
        self.predictions = np.array(predictions)
        return self.predictions

    def get_recommended_target(self, uav_x, uav_y, env):
        """Returns index of highest-priority sensor to visit next"""
        scores = []
        for i, sensor in enumerate(env.sensors):
            if i in env.dead_sensors:
                scores.append(-999)
                continue
            dist = np.sqrt((uav_x - sensor.x)**2 + (uav_y - sensor.y)**2)
            dist_score = 1.0 / (dist + 1)
            urgency = self.predictions[i] if len(self.predictions) > i else 0.5
            scores.append(dist_score * 0.4 + urgency * 0.6)
        return np.argmax(scores)
