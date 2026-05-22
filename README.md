# UAV-WSN Multi-Agent Deep Reinforcement Learning

A final year project implementing intelligent multi-UAV coordination for wireless sensor networks using LSTM-guided Deep Reinforcement Learning (PPO).

## Key Results
| System | Avg Reward | Coverage |
|--------|-----------|----------|
| Single DRL UAV | 5102 | ~80% |
| Multi-UAV DRL (2 agents) | 8293 | ~95% |
| Random Baseline | 2590 | ~40% |

**97% improvement over random baseline. 62% improvement with Multi-UAV vs Single UAV.**

## Novel Features (not in standard papers)
- Multi-UAV coordination with overlap penalty — two agents learn to divide territory
- Energy-aware sensor prioritization — UAV rescues dying sensors first
- Live real-time web dashboard (Flask) showing UAV navigation
- LSTM predictor for sensor demand forecasting

## Tech Stack
Python, PyTorch, Stable-Baselines3, Gymnasium, Flask, Matplotlib

## Project Structure
- `env/wsn_env.py` — single UAV simulation environment
- `env/multi_uav_env.py` — multi-UAV coordination environment
- `models/lstm_predictor.py` — LSTM demand forecasting
- `train_final.py` — single UAV training
- `train_multi.py` — multi-UAV training
- `evaluate_final.py` — single UAV evaluation
- `evaluate_multi.py` — multi-UAV vs single comparison
- `dashboard.py` — live Flask web dashboard

## How to Run
```bash
pip install torch stable-baselines3 gymnasium matplotlib numpy flask
python3 train_final.py       # Train single UAV
python3 train_multi.py       # Train multi-UAV
python3 evaluate_multi.py    # Compare results
python3 dashboard.py         # Launch live dashboard
```

## Reference
Based on: LSTM-Characterized Deep Reinforcement Learning for Continuous Flight Control
and Resource Allocation in UAV-Assisted Sensor Network (IEEE WSN09GK)
