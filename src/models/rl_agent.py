import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance
        self.max_steps = len(df) - 1

        # Actions: 0-Hold, 1-Buy, 2-Sell
        self.action_space = spaces.Discrete(3)

        # Observations: [RSI, MACD, Bollinger_high, SMA_20, Close]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _next_observation(self):
        obs = self.df.loc[self.current_step, ['RSI', 'MACD', 'Bollinger_high', 'SMA_20', 'Close']]
        return obs.values.astype(np.float32)

    def step(self, action):
        prev_worth = self.net_worth
        price = self.df.loc[self.current_step, 'Close'].item()  # <--- FIX HERE clearly added .item()

        # Actions clearly defined
        if action == 1 and self.balance > price:
            self.position += self.balance / price
            self.balance = 0
        elif action == 2 and self.position > 0:
            self.balance += self.position * price
            self.position = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps

        self.net_worth = self.balance + (self.position * price)
        reward = self.net_worth - prev_worth

        return self._next_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance
        return self._next_observation()

def train_rl_agent(df, model_path='rl_trading_agent'):
    env = TradingEnv(df)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save(model_path)

def load_rl_agent(df, model_path='rl_trading_agent'):
    env = TradingEnv(df)
    model = DQN.load(model_path, env=env)
    return model, env