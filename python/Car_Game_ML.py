from mss import mss
import pyautogui
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

class Game(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {'top': 190, 'left': 440, 'width': 610, 'height': 600}
        self.done_location = {'top': 190, 'left': 440, 'width': 120, 'height': 20}

    def step(self, action):
        action_map = {0: 'left', 1: 'right', 2: 'no_op'}
        if action != 2:
            pyautogui.press(action_map[action])
        done, done_cap = self.get_done()
        new_observation = self.get_observation()
        reward = 1
        info = {}
        return new_observation, reward, done, info

    def render(self):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def reset(self):
        time.sleep(1)
        pyautogui.press('left')
        return self.get_observation()

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100,83))
        channel = np.reshape(resized, (1,83,100))
        return channel

    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3].astype(np.uint8)
        done_strings = ['GAME', 'GAHE']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done, done_cap

env = Game()

# for episode in range(10): 
#     obs = env.reset()
#     done = False  
#     total_reward   = 0
#     while not done: 
#         obs, reward,  done, info =  env.step(env.action_space.sample())
#         total_reward  += reward
#     print('Total Reward for episode {} is {}'.format(episode, total_reward))   

# env_checker.check_env(env)

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose = 1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True  

CHECKPOINT_DIR = './python/script_outputs/car_train/'
LOG_DIR = './python/script_outputs/car_logs/'

callback = TrainAndLoggingCallback(check_freq = 1000, save_path = CHECKPOINT_DIR)

model = DQN('CnnPolicy', env, tensorboard_log = LOG_DIR, verbose = 1, buffer_size = 400000, learning_starts = 1000)
model.learn(total_timesteps = 100000, callback = callback)

# for episode in range(5):   
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(int(action))
#         total_reward += reward
#     print('Total Reward for episode {} is {}'.format(episode, total_reward))