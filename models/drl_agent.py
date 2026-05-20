import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from models.lstm_predictor import LSTMPredictor, SensorDataBuffer

class TrainingCallback(BaseCallback):
    """Tracks rewards during training so we can plot later"""
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []

    def _on_step(self):
        reward = self.locals.get("rewards", [0])[0]
        self.current_rewards.append(reward)
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(sum(self.current_rewards))
            self.episode_lengths.append(len(self.current_rewards))
            self.current_rewards = []
        return True

class UAVAgent:
    def __init__(self, env, n_sensors=10):
        self.env = env
        self.n_sensors = n_sensors

        # LSTM for demand prediction
        self.lstm_model = LSTMPredictor(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        self.data_buffer = SensorDataBuffer(
            n_sensors=n_sensors,
            history_len=10
        )

        # PPO agent — the DRL brain
        self.ppo_agent = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/"
        )
        self.callback = TrainingCallback()

    def train(self, total_timesteps=50000):
        print("Starting PPO training...")
        print(f"Total timesteps: {total_timesteps}")
        print("-" * 40)
        self.ppo_agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True
        )
        print("-" * 40)
        print("Training complete!")
        return self.callback

    def save(self, path="models/trained_uav_agent"):
        self.ppo_agent.save(path)
        print(f"Agent saved to {path}")

    def load(self, path="models/trained_uav_agent"):
        self.ppo_agent = PPO.load(path, env=self.env)
        print(f"Agent loaded from {path}")

    def predict(self, obs):
        action, _ = self.ppo_agent.predict(obs, deterministic=True)
        return action
