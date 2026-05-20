# UAV-Assisted Wireless Sensor Network with LSTM + Deep Reinforcement Learning

A final year project implementing an intelligent UAV navigation system for wireless sensor networks using LSTM-guided Deep Reinforcement Learning (PPO).

## Results
| Policy | Avg Reward | vs Random |
|--------|-----------|-----------|
| DRL Agent | 5102 | +97% |
| Random | 2590 | baseline |
| Greedy | 5787 | hand-coded |

## Key Features
- Custom Gymnasium simulation environment (10 sensors, 100x100 grid)
- PPO-based DRL agent that learns optimal UAV flight paths
- LSTM predictor for sensor data demand forecasting
- 97% improvement over random baseline after 200,000 training steps

## Tech Stack
Python, PyTorch, Stable-Baselines3, Gymnasium, Matplotlib

## Project Structure
- `env/` — WSN simulation environment
- `models/` — LSTM predictor and DRL agent
- `train_final.py` — training script
- `evaluate_final.py` — evaluation and visualization
- `results/` — training graphs and UAV flight path plots

## How to Run
```bash
pip install torch stable-baselines3 gymnasium matplotlib numpy
python3 train_final.py
python3 evaluate_final.py
```

## Reference
Based on: LSTM-Characterized Deep Reinforcement Learning for Continuous Flight Control
and Resource Allocation in UAV-Assisted Sensor Network (IEEE)
