import os
import numpy as np
import gymnasium as gym
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# --- your env registration function must be importable here ---
# Make sure this is the same function you used during training.
from citation_env_gauss import register_citation_env

# =============== Paths & constants ===============
LOGDIR = os.path.abspath(r".\OneDrive\Year 5\Bio-inspired\data")
LOGDIRGAUSS = os.path.abspath(r".\OneDrive\Year 5\Bio-inspired\data_gaussian")
BEST_MODEL_PATH = os.path.join(LOGDIR, "best_model.zip")
VECNORM_PATH = os.path.join(LOGDIR, "vecnormalize.pkl")
VECNORM_PATH_GAUSS =  os.path.join(LOGDIRGAUSS, "vecnormalize.pkl")
MAT_PATH = r"C:\Users\Aksha\OneDrive\Year 5\Bio-inspired\Citation_controller.mat"
ENV_ID = "Citation-nlgauss"
# MAX_EP_STEPS = 1000
TIMESTEPS = 10
EVALFREQ = 10


# =============== Load A, B and define reward fn ===============
data = loadmat(MAT_PATH)
A = data["A1"].astype(float)
B = data["B"].astype(float)

register_citation_env(env_id="Citation-nlgauss",  A=A, B=B)


def my_reward(next_state: np.ndarray, state: np.ndarray, dI: float, I_next: float,  dt: float) -> float:
    h = float(next_state[15])        
    y = float(next_state[16])    
    # penalties
    # w_ay = 0.01                           
    w_h  = 1e-4
    w_x = 0.05
    w_y = 0.06
    w_vh = 0.015
    w_i = 0.75
    w_icurr = 0.95
    

   

    # penalty_ay = w_ay * abs(ay)
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


    #gradient following and low I preference
    # r_Ichange =  w_i * max(0.0, -dI) - w_i*1.1 * max(0.0, dI) #increased punishment for dI increase
    r_Ichange =  - w_i * max(0.0, dI) #increased punishment for dI increase
    r_Icurr = w_icurr * -I_next
    

    #vertical rate cost
    h_new = float(next_state[15])
    h_old = float(state[15])
    v_h = (h_new - h_old)/dt  
    vh  = np.clip(v_h, -20, 20)
    r_rate = -w_vh * abs(vh)

    

    #followthrough

    delx = float(next_state[14] - state[14])   
    r_delx = w_x * np.tanh(delx / 15.0) * np.exp(- I_next/0.75)

    #lateral
    penalty_y = w_y * max(0.0, abs(y) - 50.0)


    return r_delx - penalty_h + oobpenalty + pitch_term - penalty_y + r_rate + r_Ichange + r_Icurr



def make_train_env():
    return gym.make(ENV_ID, A = A, B = B, render_mode = None, reward_fn = my_reward)

def make_eval_env():
    return gym.make(ENV_ID, A = A, B = B, render_mode = None, reward_fn = my_reward)

def make_test_env(): 
    return gym.make(ENV_ID, A = A, B = B, render_mode = "human", reward_fn = my_reward)



raw_train = DummyVecEnv([make_train_env])
venv = VecNormalize.load(VECNORM_PATH, raw_train)
venv.training = True
venv.norm_reward = True

raw_eval = DummyVecEnv([make_eval_env])
evalenv = VecNormalize.load(VECNORM_PATH, raw_eval)
evalenv.training = False
evalenv.norm_reward = False
evalenv.obs_rms = venv.obs_rms

model = SAC.load(BEST_MODEL_PATH, env = venv, device = "auto", print_system_info= False, gamma = 0.99)
print("Current discount factor:", model.gamma)



eval_cb = EvalCallback(
    evalenv,
    eval_freq=EVALFREQ,
    n_eval_episodes=10,
    deterministic=True,
    best_model_save_path=LOGDIRGAUSS,
    log_path=LOGDIRGAUSS,
)



start = time.time()
model.learn(total_timesteps = TIMESTEPS, progress_bar=True, log_interval=10, callback = eval_cb, reset_num_timesteps= False)
venv.save(os.path.join(".\OneDrive\Year 5\Bio-inspired\data_gaussian", "vecnormalize.pkl"))
print(f"Training seconds: {time.time()-start:.1f}")

venv.save(VECNORM_PATH_GAUSS)


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
