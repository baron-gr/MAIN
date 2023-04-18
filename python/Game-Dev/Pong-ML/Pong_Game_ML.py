from mss import mss
import pyautogui
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import datetime
import time
from gym import Env
from gym.spaces import Box, Discrete
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
from sklearn.model_selection import ParameterGrid

class PyGame(Env):
    def __init__(self): 
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {'top': 230, 'left': 30, 'width': 700, 'height': 505}
        self.done_location = {'top': 450, 'left': 230, 'width': 280, 'height': 70}

    def step(self, action):
        action_map = {0: 'up', 1: 'down', 2: 'no_op'}
        if action != 2:
            pyautogui.press(action_map[action])
        res,  done, done_cap = self.get_done()
        new_observation = self.get_observation()
        reward += 1
        info = {}
        return new_observation, reward, done, info

    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def reset(self):
        time.sleep(1)
        pyautogui.press('space')
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
        return res, done, done_cap
    
env = PyGame()

## Train Model
# class TrainAndLoggingCallback(BaseCallback):
#     def __init__(self, check_freq, save_path, verbose = 1):
#         super(TrainAndLoggingCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.save_path = save_path

#     def _init_callback(self):
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok = True)

#     def _on_step(self):
#         if self.n_calls % self.check_freq == 0:
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             model_path = os.path.join(self.save_path, f'best_model_{timestamp}_{self.n_calls}')
#             self.model.save(model_path)
#         return True

# CHECKPOINT_DIR = '/Users/bgracias/ML-Models/Pong-Models/train/'
# LOG_DIR = '/Users/bgracias/ML-Models/Pong-Models/logs/'

# callback = TrainAndLoggingCallback(check_freq = 1000, save_path = CHECKPOINT_DIR)

# params_grid = {
#     'learning_rate': [0.00025, 0.0005, 0.001],
#     'buffer_size': [70000, 300000, 800000],
#     'batch_size': [32, 64, 128]
# }

# best_reward = float('-inf')
# best_params = None

# for params in ParameterGrid(params_grid):
#     model = DQN('CnnPolicy', env, tensorboard_log = LOG_DIR,
#                 learning_starts = 1000,
#                 exploration_fraction = 0.1,
#                 exploration_initial_eps = 1.0,
#                 exploration_final_eps = 0.02,
#                 train_freq = 4,
#                 target_update_interval = 1000,
#                 verbose = 0,
#                 **params)
    
#     mean_reward = 0
#     for i in range(10):
#         obs = env.reset()
#         done = False
#         while not done:
#             action, _ = model.predict(obs)
#             obs, reward, done, info = env.step(action)
#             mean_reward += reward
#     mean_reward /= 10
    
#     print('Parameters:', params)
#     print('Mean reward:', mean_reward)
    
#     if mean_reward > best_reward:
#         best_reward = mean_reward
#         best_params = params

# print('Best parameters:', best_params)
# print('Best mean reward:', best_reward)

# final_model = DQN('CnnPolicy', env, tensorboard_log = LOG_DIR,
#                   learning_starts = 1000,
#                   exploration_fraction = 0.1,
#                   exploration_initial_eps = 1.0,
#                   exploration_final_eps = 0.02,
#                   train_freq = 4,
#                   target_update_interval = 1000,
#                   verbose = 1,
#                   **best_params)

# final_model.learn(total_timesteps = 150000, callback = callback)

## Testing the model
## Comment out previous bits
# model = DQN.load(os.path.join('train', 'best_model_20230418_055131_100000.zip'))

# for episode in range(10):   
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(int(action))
#         total_reward += reward
#     print('Total Reward for episode {} is {}'.format(episode, total_reward))

## Game obs
# plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_BGR2RGB))
# plt.show()

## Game over obs
# res, done, done_cap = env.get_done()
# print(done)
# print(res)
# plt.imshow(done_cap)
# plt.show()

## Score obs
# score, score_cap = env.get_score()
# print(score)
# plt.imshow(score_cap)
# plt.show()

# Testing environment
# for episode in range(5): 
#     obs = env.reset()
#     done = False  
#     total_reward = 0
#     while not done:  
#         obs, reward, done, info =  env.step(env.action_space.sample())
#         total_reward += reward
#     print('Total Reward for episode {} is {}'.format(episode, total_reward))