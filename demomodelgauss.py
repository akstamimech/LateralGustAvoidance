
# You must have saved vecnormalize.pkl, as well as best_model.zip to run this file


import os
import numpy as np
import gymnasium as gym
from scipy.io import loadmat
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from citation_env_gauss import register_citation_env

LOGDIR = os.path.abspath(r".\OneDrive\Year 5\Bio-inspired\data_gaussian")
BEST_MODEL_PATH = os.path.join(LOGDIR, "best_model.zip")
VECNORM_PATH = os.path.join(LOGDIR, "vecnormalize.pkl")
MAT_PATH = r"C:\Users\Aksha\OneDrive\Year 5\Bio-inspired\Citation_controller.mat"
ENV_ID = "Citation-nlgauss"
# MAX_EP_STEPS = 1000

data = loadmat(MAT_PATH)
A = data["A1"].astype(float)
B = data["B"].astype(float)




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
    r_Ichange =  w_i * max(0.0, -dI) - w_i*1.1 * max(0.0, dI) 
    r_Icurr = w_icurr * -I_next
    

    #vertical rate cost
    h_new = float(next_state[15])
    h_old = float(state[15])
    v_h = (h_new - h_old)/dt  
    vh  = np.clip(v_h, -20, 20)
    r_rate = -w_vh * abs(vh)

    

    #followthrough

    delx = float(next_state[14] - state[14])   
    r_delx = w_x * np.tanh(delx / 15.0) * np.exp(- I_next)

    #lateral
    penalty_y = w_y * max(0.0, abs(y) - 50.0)


    return r_delx - penalty_h + oobpenalty + pitch_term - penalty_y + r_rate + r_Ichange + r_Icurr


register_citation_env(env_id=ENV_ID, A=A, B=B)

def make_env_base(render_mode=None):
    """Create base env with identical kwargs as training; add TimeLimit to ensure termination."""
    env = gym.make(ENV_ID, A=A, B=B, render_mode=render_mode, reward_fn=my_reward)
    return env


assert os.path.exists(BEST_MODEL_PATH), f"best model not found: {BEST_MODEL_PATH}"
assert os.path.exists(VECNORM_PATH), f"VecNormalize stats not found: {VECNORM_PATH} (did you save it after training?)"

print("Using:")
print("  BEST_MODEL_PATH:", BEST_MODEL_PATH)
print("  VECNORM_PATH   :", VECNORM_PATH)


model = SAC.load(BEST_MODEL_PATH)



NUM_EPISODES = 1 #reduced from 15
total_reward = 0.0
reward_list = []

base_vec = DummyVecEnv([lambda: make_env_base(render_mode=None)])
test_env = VecNormalize.load(VECNORM_PATH, base_vec)  # loads obs_rms, clip settings
test_env.training = False
test_env.norm_reward = False   
model.set_env(test_env)


for episode in range(NUM_EPISODES):
    obs = test_env.reset()
    done = [False]
    ep_reward = 0.0
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        ep_reward += float(reward[0])
    print(f"[Vectorized test] Episode reward (true scale): {ep_reward:.6f}")
    total_reward +=ep_reward
    reward_list.append(ep_reward)

total_reward_ave = total_reward/NUM_EPISODES
std = np.std(reward_list)

print(std)
print(reward_list)


human_vec = DummyVecEnv([lambda: make_env_base(render_mode="human")])


human_vec = VecNormalize(human_vec, norm_obs=True, norm_reward=False, clip_obs=test_env.clip_obs)
human_vec.obs_rms = test_env.obs_rms 
human_vec.training = False


model.set_env(human_vec)
obs = human_vec.reset()
done = [False]
x_path = []
h_path = []
y_path = []
theta_path = []
ail_path = []
rud_path = []
elevator_path = []

while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = human_vec.step(action)
    raw_state = human_vec.get_attr("state", indices = [0])[0]
    action = human_vec.get_attr("u", indices= [0])[0]
    x_path.append(raw_state[14])
    h_path.append(raw_state[15])
    y_path.append(raw_state[16])
    theta_path.append(raw_state[12])
    ail_path.append(action[0])
    rud_path.append(action[1])
    elevator_path.append(action[5])



while plt.get_fignums():
    plt.pause(0.1)
human_vec.close()
test_env.close()



x = np.asarray(x_path)
y = np.asarray(y_path)
z = np.asarray(h_path)        
theta = np.asarray(theta_path)
dela = np.asarray(ail_path)
delr = np.asarray(rud_path)
dele = np.asarray(elevator_path)

fig, axes = plt.subplots(3, 2, figsize=(12, 8))


axes[0,0].plot(x[:-1], y[:-1], linewidth=1.5)
axes[0,0].axhline(50, linestyle='--', linewidth=1)
axes[0,0].axhline(-50, linestyle='--', linewidth=1)
axes[0,0].set_xlabel('x (meters)')
axes[0,0].set_ylabel('y (meters)')
axes[0,0].set_title('Lateral position y vs x')
axes[0,0].grid(True, linewidth=0.3)


axes[0,1].plot(x[:-1], theta[:-1], linewidth=1.5)
axes[0,1].set_xlabel('x (meters)')
axes[0,1].set_ylabel(r'$\theta$ (rad)')
axes[0,1].set_title(r'Pitch $\theta$ vs x')
axes[0,1].grid(True, linewidth=0.3)


axes[1,0].plot(x[:-1], z[:-1], linewidth=1.5)
axes[1,0].set_xlabel('x (meters)')
axes[1,0].set_ylabel('z (meters)')
axes[1,0].set_title('Altitude z vs x')
axes[1,0].grid(True, linewidth=0.3)


axes[1,1].plot(x[:-1], dela[:-1], linewidth=1.5)
axes[1,1].set_xlabel('x (meters)')
axes[1,1].set_ylabel('del a')
axes[1,1].set_title('Aileron deflection vs x')
axes[1,1].grid(True, linewidth=0.3)

axes[2,0].plot(x[:-1], delr[:-1], linewidth=1.5)
axes[2,0].set_xlabel('x (meters)')
axes[2,0].set_ylabel('del r')
axes[2,0].set_title('rudder deflection vs x')
axes[2,0].grid(True, linewidth=0.3)

axes[2,1].plot(x[:-1], delr[:-1], linewidth=1.5)
axes[2,1].set_xlabel('x (meters)')
axes[2,1].set_ylabel('del e')
axes[2,1].set_title('elevator deflection vs x')
axes[2,1].grid(True, linewidth=0.3)


fig.tight_layout()
plt.show()



