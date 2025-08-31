# main.py
import os
import numpy as np
from scipy.io import loadmat
import gymnasium as gym
from citation_env import register_citation_env
from stable_baselines3 import SAC 
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import time
import imageio.v2 as imageio
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt

data = loadmat('C:/Users/Aksha/OneDrive/Year 5/Bio-inspired/Citation_controller.mat')
A = data['A1'].astype(float)
B = data['B'].astype(float)

register_citation_env(env_id="Citation-nl", A=A, B=B)




def make_render_env():
    return gym.make("Citation-nl", A=A, B=B, render_mode="human")


def my_reward(next_state: np.ndarray, state: np.ndarray, dI: float, dt: float) -> float:
    h = float(next_state[15])        
    y = float(next_state[16])    
    # penalties
    # w_ay = 0.01                           
    w_h  = 1e-4
    w_x = 0.1
    w_y = 0.06
    w_vh = 0.015
    w_i = 0.95
    

   

   
    #altitude penalty
    penalty_h = w_h * max(0.0, abs(h) - 75.0)**2

    #out of bounds
    oob = (next_state[15] > 200) or (next_state[15] < -200) or (next_state[16] > 200) or (next_state[16] < -200)

    if oob == True: 
        oobpenalty = -100
    else: 
        oobpenalty = 0


    #pitch extremeness
    theta = float(next_state[12])
    a = abs(theta)

    FREE, MID = 0.15, 0.25       
    R_SMALL   = 0.01            
    W_MID     = 0.01             
    W_LARGE   = 0.10             

    if a <= FREE:
       
        pitch_term = +R_SMALL * (1.0 - a / FREE)
    elif a <= MID:
        
        pitch_term = -W_MID * (a - FREE)
    else:
        
        pitch_term = -W_MID * (MID - FREE) - W_LARGE * (a - MID)


    #gradient following and low sigma pref
    r_Ichange =  w_i * max(0.0, -dI)
    
   
    
    

    #vertical rate cost
    h_new = float(next_state[15])
    h_old = float(state[15])
    v_h = (h_new - h_old)/dt  
    vh  = np.clip(v_h, -20, 20)
    r_rate = -w_vh * abs(vh)

    

    #followthrough

    delx = float(next_state[14] - state[14])   
    r_delx = w_x * delx

    #lateral
    penalty_y = w_y * max(0.0, abs(y) - 50.0)


    return r_delx - penalty_h + oobpenalty + pitch_term - penalty_y + r_rate + r_Ichange


# def make_render_env_rgb():
#     return gym.make("Citation-v0", A=A, B=B, render_mode="rgb_array")
def make_train_env():
    return gym.make("Citation-nl", A = A, B = B, render_mode = None, reward_fn = my_reward)

def make_eval_env():
    return gym.make("Citation-nl", A = A, B = B, render_mode = None, reward_fn = my_reward)

def make_test_env(): 
    return gym.make("Citation-nl", A = A, B = B, render_mode = "human", reward_fn = my_reward)



# train_env = gym.make("Citation-v0", A = A, B = B, render_mode = None, reward_fn = my_reward)

venv = DummyVecEnv([make_train_env])
venv = VecNormalize(venv, norm_obs = True, norm_reward = True, clip_obs = 10.0)

model = SAC(policy="MlpPolicy", env=venv, verbose=1, seed=42)

evalenv = DummyVecEnv([make_eval_env])
evalenv = VecNormalize(evalenv, norm_obs = True, norm_reward = False, clip_obs = 10.0)
evalenv.obs_rms = venv.obs_rms
evalenv.training = False

eval_cb = EvalCallback(
    evalenv,
    eval_freq=10000,      
    n_eval_episodes=10,     
    deterministic=True,
    render=False,
    best_model_save_path = '.\OneDrive\Year 5\Bio-inspired\data',
    log_path = '.\OneDrive\Year 5\Bio-inspired\data'
)

start = time.time()
model.learn(total_timesteps = 200000, progress_bar=True, log_interval=10, callback = eval_cb)
venv.save(os.path.join(".\OneDrive\Year 5\Bio-inspired\data", "vecnormalize.pkl"))
print(f"Training seconds: {time.time()-start:.1f}")


test_env = DummyVecEnv([make_test_env])
test_env = VecNormalize(test_env, norm_obs = True, norm_reward = True, clip_obs = 10.0)

# test_env.seed(2025)

obs = test_env.reset()
done = False
ep_reward = 0.0

x_path = []
h_path = []
y_path = []
while not done:
    action, _ = model.predict(obs, deterministic=True)  
    obs, reward, done, info = test_env.step(action)
    raw_state = test_env.get_attr("state", indices = [0])[0]
    # ns = test_env.get_attr("next_state", indices = [0])[0]
    x_path.append(raw_state[14])
    h_path.append(raw_state[15])
    y_path.append(raw_state[16])
    ep_reward += float(reward)





test_env.close()
print("Test episode reward:", ep_reward)
print(y_path[:-1])
plt.plot(x_path[:-1], h_path[:-1])
plt.show()


