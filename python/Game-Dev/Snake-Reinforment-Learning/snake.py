import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class Snake_game(gym.Env):
    metadata = {'render.modes': ['console','rgb_array']}
    n_actions = 3 #3 possible steps each turn
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    #Grid label constants
    EMPTY = 0
    SNAKE = 1
    WALL = 2
    FOOD = 3
    
    #Rewards
    REWARD_WALL_HIT = -20
    REWARD_PER_STEP_TOWARDS_FOOD = 1
    REWARD_PER_FOOD = 50
    MAX_STEPS_AFTER_FOOD = 200 #stop if we go too long without food to avoid infinite loops

    def __init__(self, grid_size=12):
        super(Snake_game, self).__init__()
        #Steps so far
        self.stepnum = 0; self.last_food_step=0
        # Size of the 2D grid (including walls)
        self.grid_size = grid_size
        # Initialize the snake
        self.snake_coordinates = [(1,1), (2,1)]
        #Init the grid
        self.grid = np.zeros( (self.grid_size, self.grid_size) ,dtype=np.uint8) + self.EMPTY
        self.grid[0,:] = self.WALL; self.grid[:,0] = self.WALL; #wall at the egdes
        self.grid[int(grid_size/2),3:(grid_size-3)] = self.WALL; #inner wall to make the game harder
        self.grid[4:(grid_size-4),int(grid_size/2-1)] = self.WALL; #inner wall to make the game harder
        #self.grid[int(grid_size/2),2:(grid_size-2)] = self.WALL; #inner wall to make the game harder
        self.grid[self.grid_size-1,:] = self.WALL; self.grid[:,self.grid_size-1] = self.WALL
        for coord in self.snake_coordinates:
            self.grid[ coord ] = self.SNAKE  #put snake on grid
        self.grid[3,3] = self.FOOD  #Start in upper right corner
        #Init distance to food
        self.head_dist_to_food = self.grid_distance(self.snake_coordinates[-1],np.argwhere(self.grid==self.FOOD)[0])
        #Store init values
        self.init_grid = self.grid.copy()
        self.init_snake_coordinates = self.snake_coordinates.copy()
        
        # The action space
        self.action_space = spaces.Discrete(self.n_actions)
        # The observation space, "position" is the coordinates of the head; "direction" is which way the sanke is heading, "grid" contains the full grid info
        self.observation_space = gym.spaces.Dict(
            spaces={
                "position": gym.spaces.Box(low=0, high=(self.grid_size-1), shape=(2,), dtype=np.int32),
                "direction": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
                "grid": gym.spaces.Box(low = 0, high = 3, shape = (self.grid_size, self.grid_size), dtype=np.uint8),
            })

    def grid_distance(self,pos1,pos2):
        return np.linalg.norm(np.array(pos1,dtype=np.float32)-np.array(pos2,dtype=np.float32))

    def reset(self):
        # Reset to initial positions
        self.stepnum = 0; self.last_food_step=0
        self.grid = self.init_grid.copy()
        self.snake_coordinates = self.init_snake_coordinates.copy()
        #Init distance to food
        self.head_dist_to_food = self.grid_distance(self.snake_coordinates[-1],np.argwhere(self.grid==self.FOOD)[0] )
        return self._get_obs()
             
    def _get_obs(self):
            direction = np.array(self.snake_coordinates[-1]) - np.array(self.snake_coordinates[-2])
            #return observation in the format of self.observation_space
            return {"position": np.array(self.snake_coordinates[-1],dtype=np.int32),
                    "direction" : direction.astype(np.int32),
                    "grid": self.grid}
                         
    def step(self, action):
        #Get direction for snake
        direction = np.array(self.snake_coordinates[-1]) - np.array(self.snake_coordinates[-2])
        if action == self.STRAIGHT:
            step = direction #step in the direction the snake faces
        elif action == self.RIGHT:
            step = np.array( [direction[1], -direction[0]] )  #turn right
        elif action == self.LEFT:
            step = np.array( [-direction[1], direction[0]] )   #turn left
        else:
            raise ValueError("Action=%d is not part of the action space"%(action))
        #New head coordinate
        new_coord = (np.array(self.snake_coordinates[-1]) + step).astype(np.int32)
        #grow snake
        self.snake_coordinates.append( (new_coord[0],new_coord[1]) ) #convert to tuple so we can use it to index
        
        #Check what is at the new position
        new_pos = self.snake_coordinates[-1]
        new_pos_type = self.grid[new_pos]
        self.grid[new_pos] = self.SNAKE #this position is now occupied by the snake
        done = False; reward = 0 #by default the game goes on and no reward   
        if new_pos_type == self.FOOD:
            reward += self.REWARD_PER_FOOD
            self.last_food_step = self.stepnum
            #Put down a new food item
            empty_tiles = np.argwhere(self.grid==self.EMPTY)
            if len(empty_tiles):
                new_food_pos=empty_tiles[np.random.randint(0,len(empty_tiles))]
                self.grid[new_food_pos[0],new_food_pos[1]] = self.FOOD
            else:
                done = True #no more tiles to put the food to
        else:
            #If no food was eaten we remove the end of the snake (i.e., moving not growing)
            self.grid[ self.snake_coordinates[0] ] = self.EMPTY
            self.snake_coordinates = self.snake_coordinates[1:]
            if  (new_pos_type == self.WALL) or (new_pos_type == self.SNAKE):
                done = True #stop if we hit the wall or the snake
                reward += self.REWARD_WALL_HIT #penalty for hitting walls/tail

        #Update distance to food and reward if closer
        head_dist_to_food_prev = self.head_dist_to_food
        self.head_dist_to_food = self.grid_distance( self.snake_coordinates[-1],np.argwhere(self.grid==self.FOOD)[0] )
        if head_dist_to_food_prev > self.head_dist_to_food:
            reward += self.REWARD_PER_STEP_TOWARDS_FOOD #reward for getting closer to food
        elif head_dist_to_food_prev < self.head_dist_to_food:
            reward -= self.REWARD_PER_STEP_TOWARDS_FOOD #penalty for getting further
        
        #Stop if we played too long without getting food
        if ( (self.stepnum - self.last_food_step) > self.MAX_STEPS_AFTER_FOOD ): 
            done = True    
        self.stepnum += 1

        return  self._get_obs(), reward, done, {}
    
    def render(self, mode='rgb_array'):
        if mode == 'console':
            print(self.grid)
        elif mode == 'rgb_array':
            return self.snake_plot()
        else:
            raise NotImplementedError()

    def close(self):
        pass
    
    def snake_plot(self, plot_inline=False):
        wall_ind = (self.grid==self.WALL)
        snake_ind = (self.grid==self.SNAKE)
        food_ind = (self.grid==self.FOOD)
        #Create color array for plot, default white color
        Color_array=np.zeros((self.grid_size,self.grid_size,3),dtype=np.uint8)+255 #default white
        Color_array[wall_ind,:]= np.array([0,0,0]) #black walls
        Color_array[snake_ind,:]= np.array([0,0,255]) #bluish snake
        Color_array[food_ind,:]= np.array([0,255,0]) #green food  
        #plot
        if plot_inline:
            fig=plt.figure()
            plt.axis('off')
            plt.imshow(Color_array, interpolation='nearest')
            plt.show()
        return Color_array

#Manual testing
# import matplotlib.animation as animation
# from time import sleep

# env = Snake_game()
# env.reset()

#Image for initial state
# fig, ax = plt.subplots(figsize=(6,6))
# plt.imshow(env.render(mode='rgb_array'))
# plt.axis('off')
# plt.show()

# #Framework to save animgif
# frames = []
# fps=24

# n_steps = 20
# for step in range(n_steps):
#     print("Step {}".format(step + 1))
#     obs, reward, done, info = env.step(0)
#     print('position=', obs['position'], 'direction=', obs['direction'])
#     print('reward=', reward, 'done=', done)
#     frames.append([ax.imshow(env.render(mode='rgb_array'), animated=True)])
#     if done:
#         print("Game over!", "reward=", reward)
#         break

# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) #to remove white bounding box        
# anim = animation.ArtistAnimation(fig, frames, interval=int(1000/fps), blit=True,repeat_delay=1000)
# anim.save("snake_test.gif",dpi=150)

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

#Logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# Instantiate the env
env = Snake_game()
# wrap it
env = Monitor(env, log_dir)

#Callback, this built-in function will periodically evaluate the model and save the best version
eval_callback = EvalCallback(env, best_model_save_path='./log/',
                             log_path='./log/', eval_freq=5000,
                             deterministic=False, render=False)

# import time
# from stable_baselines3 import PPO

# #Train the agent
# max_total_step_num = 1e6

# def learning_rate_schedule(progress_remaining):
#     start_rate = 0.0001 #0.0003
#     #Can do more complicated ones like below
#     #stepnum = max_total_step_num*(1-progress_remaining)
#     #return 0.003 * np.piecewise(stepnum, [stepnum>=0, stepnum>4e4, stepnum>2e5, stepnum>3e5], [1.0,0.5,0.25,0.125 ])
#     return start_rate * progress_remaining #linearly decreasing

# PPO_model_args = {
#     "learning_rate": learning_rate_schedule, #decreasing learning rate #0.0003 #can be set to constant
#     "gamma": 0.99, #0.99, discount factor for futurer rewards, between 0 (only immediate reward matters) and 1 (future reward equivalent to immediate), 
#     "verbose": 0, #change to 1 to get more info on training steps
#     #"seed": 137, #fixing the random seed
#     "ent_coef": 0.0, #0, entropy coefficient, to encourage exploration
#     "clip_range": 0.2 #0.2, very roughly: probability of an action can not change by more than a factor 1+clip_range
# }
# starttime = time.time()
# model = PPO('MultiInputPolicy', env,**PPO_model_args)
# #Load previous best model parameters, we start from that
# if os.path.exists("log/best_model.zip"):
#     model.set_parameters("log/best_model.zip")
# model.learn(max_total_step_num,callback=eval_callback)
# dt = time.time()-starttime
# print("Calculation took %g hr %g min %g s"%(dt//3600, (dt//60)%60, dt%60) )