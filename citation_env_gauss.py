from __future__ import annotations 
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, Callable
import matplotlib.pyplot as plt
from collections import deque
from gymnasium.envs.registration import register, registry





class CitationEnvGauss(gym.Env):



    metadata = {"render_modes": ['human', 'rgb_array'], "render_fps": 1}

    V_trim = 59.9
    alpha_trim = 0.0
    theta_trim = 0.0
    beta_trim = 0.0
    b = 13.36
    tau1 = 0.0991
    tau2 = 0.5545
    tau3 = 0.4159
    tau4 = 0.0600
    tau5 = 0.3294
    tau6 = 0.2243
    Lg = 150.0



    def __init__(self, A: np.ndarray, B: np.ndarray, *, dt: float = 0.05, max_steps: int = 10000,
                  x0: float = 0.0, h0 = 0.0, render_mode: Optional[str] = None, 
                  reward_fn: Optional[Callable[[np.ndarray, float], float]] = None,
                  rng_seed: Optional[int] = 123, sigma_config: Dict[str, Any] = None):  

        super().__init__()

        self.A = A 
        self.B_base = B 
        self.dt = dt 
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng(rng_seed)


        self.i_u = 10
        self.i_alpha = 11
        self.i_theta = 12
        self.beta_row = 0
        self.yawrate = 3

        self.h_list = deque(maxlen = 3)
        self.rmsaylist = deque(maxlen = 3)
        self.I_list = deque(maxlen=3)
        self.ayg_hist = deque(maxlen=int(3.0/self.dt))  # 3s window
        self.I_prev = 0.0

        self._just_reset = True

        if sigma_config is None: 
            sigma_config = {}
        
        self._build_sigma_field(xlim= (0, 12000),
            zlim= (0 - 200.0, 0+ 200.0), #hardcoded x0, h0, why is it not working
            nx= 1200,
            nz= 400,
            amplitude= None,
            wavelength= None,
            width= None,
            sigma_max= 4.0,
            sigma_min= 0.0,
            ngaussmin=80, ngaussmax=100,
        )


        #---------------observed states---------------------------------

        '''
        observation space: continuous values, 16 states all continuous + ay
        '''

        high_obs = np.full(17, 1e6, dtype = np.float32)
        high_obs[15] = 201
        high_obs[16] = 201
        self.observation_space = spaces.Box(low = -high_obs, high=high_obs, dtype = np.float32) #17 states with limits high_obs


        #----------------action space--------------------------------

        '''
        action space: continuous values, for now only u_cmd[0,1,5], elevator deflection, + white noise values??
        '''

        act_high = np.array([0.2, 0.2, 0.2]) #aileron, rudder, elevator
        self.action_space = spaces.Box(low = -act_high, high = act_high, dtype = np.float32) 

        self.state = None
        self.u = None
        self.steps = 0 

        self._fig = None
        self._ax = None
        self._traj_x = []
        self._traj_h = []
        self._traj_y = [] 


        # ---------- helpers to size sigma field wrt horizon ----------
    def _xmax(self, steps: int, dt: float) -> float:
        return self.V_trim * ((steps + 1) * dt)

    def _padx(self, steps: int, dt: float) -> float:
        return 0.1 * self._xmax(steps, dt)



    def _build_sigma_field(self, xlim, zlim, nx, nz, amplitude = None, wavelength = None, width = None, 
                           sigma_max = 4.0, sigma_min = 0.0, ngaussmin = 50, ngaussmax = 55, 
                           anisotropic = False, peak_mode = "uniform", peak_fixed = None): 
        
        x = np.linspace(*xlim, nx)
        z = np.linspace(*zlim, nz)
        X, Z = np.meshgrid(x, z)

            #create ngauss gaussians, with randomized stdev and means (x,z)
            #sigmameansforgaussians = random sigma within a reasonable range, clipped between sigma min and sigma max
            #sigmastdevsforgaussians = random stdevs within a reasonable range, clipped between stdevmin and stvdevmax
            # meanlist = [] each element is (1x2)
            # stdevlist = [] 
            # gaussarr = initialized array of (x,z) matrix within 0 everywhere
            #ngauss = random number between ngaussmin and ngaussmax
            #for i in range(ngauss): 
            #   mean = (random mean couple within (x,z))
            #   stdev = rangom stdev within stdevmin and stdevmax
            #   create a gaussian distribution with (mean,stdev) = array of (x,z) values = currentgauss
            #   gaussarr = gaussarr + currentgauss

        rng = getattr(self, "np_random", np.random)

        Lx = xlim[1] - xlim[0]
        Lz = zlim[1] - zlim[0]

        field = np.full_like(X, float(sigma_min), dtype = float)
        ngauss = int(rng.integers(ngaussmin, ngaussmax + 1))

        for _ in range(ngauss):
            mx = rng.uniform(xlim[0], xlim[1])
            mz = rng.uniform(zlim[0], zlim[1])

            sx = self.rng.uniform(0.03*Lx, 0.05*Lx)
            sz = self.rng.uniform(0.06*Lz, 0.1*Lz)

            peak = rng.uniform(sigma_min, sigma_max)

            G = peak * np.exp(-0.5 * (((X - mx) / sx) ** 2 + ((Z - mz) / sz) ** 2))
            field += G

        sigma = np.clip(field, sigma_min, sigma_max)
        
        # print(f"field = {field}")
        self.X, self.Z, self.sigma = X, Z, sigma
        self.xlim, self.zlim = xlim, zlim
        self.nx, self.nz = nx, nz
        self.x0, self.z0 = xlim[0], zlim[0]
        self.dx = (xlim[1]-xlim[0])/(nx-1)
        self.dz = (zlim[1]-zlim[0])/(nz-1)



    def _sigma_at_point(self, xq: float, zq: float) -> float:
        ix = int(np.clip(round((xq - self.x0)/self.dx), 0, self.nx-1))
        iz = int(np.clip(round((zq - self.z0)/self.dz), 0, self.nz-1))
        return float(self.sigma[iz, ix])

    # ---------- turbulence-injected B ----------
    def _create_B_turbed(self, B: np.ndarray, sigma_val: float) -> np.ndarray:
        V_trim = self.V_trim
        tau1,tau2,tau3,tau4,tau5,tau6 = self.tau1,self.tau2,self.tau3,self.tau4,self.tau5,self.tau6
        Lg = self.Lg

        sigmavg   = sigma_val
        sigmaug_V = 0.0
        sigmabg   = sigmavg / V_trim
        sigmaag   = sigma_val / V_trim

        Iug0 = 0.0249 * sigmaug_V**2
        Iag0 = 0.0182 * sigmaag**2

        bug1 = tau3 * np.sqrt(Iug0 * (V_trim/Lg)) / (tau1 * tau2)
        bug2 = (1 - tau3*(tau1 + tau2)/(tau1 * tau2)) * np.sqrt(Iug0 * (V_trim/Lg)**3) / (tau1 * tau2)
        bag1 = tau6 * np.sqrt(Iag0 * V_trim/Lg) / (tau4 * tau5)
        bag2 = (1 - tau6*(tau4 + tau5)/(tau4 * tau5)) * np.sqrt(Iag0 * (V_trim/Lg)**3) / (tau4 * tau5)
        bbg1 = sigmabg * np.sqrt(3 * V_trim/Lg)
        bbg2 = (1 - 2*np.sqrt(3)) * sigmabg * np.sqrt((V_trim/Lg)**3)

        B_turbed = B.copy()
        B_turbed[4,2] = bug1
        B_turbed[5,2] = bug2
        B_turbed[6,3] = bag1
        B_turbed[7,3] = bag2
        B_turbed[8,4] = bbg1
        B_turbed[9,4] = bbg2
        return B_turbed
    


    # ---------- dynamics ----------
    def _linear_system(self, z: np.ndarray, B: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        return self.A @ z + B @ u_cmd

    def _kinematics_xh(self, z: np.ndarray) -> Tuple[float, float]:
        u_rel      = z[self.i_u]
        alpha_rel  = z[self.i_alpha]
        theta_rel  = z[self.i_theta]
        beta_rel = z[self.beta_row]

        u_abs     = self.V_trim + u_rel
        alpha_abs = self.alpha_trim + alpha_rel
        theta_abs = self.theta_trim + theta_rel
        beta_abs = self.beta_trim + beta_rel

        w_abs = u_abs * np.tan(alpha_abs)

        v_abs = u_abs * np.tan(beta_abs)


        xdot  = np.cos(theta_abs)*u_abs - np.sin(theta_abs)*w_abs
        hdot  = np.sin(theta_abs)*u_abs + np.cos(theta_abs)*w_abs
        ydot = v_abs #check
        return float(xdot), float(hdot), float(ydot)

    # def _lateral_acceleration(self, z: np.ndarray, B: np.ndarray, u_cmd: np.ndarray) -> float:
    #     betadot = self.A[self.beta_row, :] @ z + B[0, :] @ u_cmd
    #     r = (2 * self.V_trim / self.b) * z[self.yawrate]
    #     ay = self.V_trim * (betadot + r)
    #     puregustay = ay - 9.8 * np.sin(z[1]) - self.V_trim *r
    #     rms = np.sqrt(puregustay**2)

    #     return float(rms) #edited pure gut ay only 
    
    def _lateral_acceleration(self, z, B, u_cmd) -> float:
        betadot = self.A[self.beta_row, :] @ z + B[0, :] @ u_cmd
        r = (2.0 * self.V_trim / self.b) * z[self.yawrate]
        ay = self.V_trim * (betadot + r)
        phi = z[1]  # roll index
        ay_gust = ay - 9.81*np.sin(phi) - self.V_trim*r  
        return float(ay_gust)
        
        

    def _f_full(self, state: np.ndarray, B: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        z = state[:14]
        xdot, hdot, ydot = self._kinematics_xh(z)
        zdot = self._linear_system(z, B, u_cmd)
        return np.concatenate([zdot, [xdot, hdot, ydot]])

    def _rk4_step(self, state: np.ndarray, B: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        dt = 0.1
        k1 = self._f_full(state, B, u)
        k2 = self._f_full(state + 0.5*dt*k1, B, u)
        k3 = self._f_full(state + 0.5*dt*k2, B, u)
        k4 = self._f_full(state + dt*k3, B, u)
        return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    

    # ---------- working with gymnasium ------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._build_sigma_field(xlim= (0 - self._padx(self.max_steps, self.dt), 0 + self._xmax(self.max_steps, self.dt) + self._padx(self.max_steps, self.dt)),
            zlim= (0 - 200.0, 0 + 200.0), #hardcodig x0, h0
            nx= 2400,
            nz= 400,
            amplitude= None,
            wavelength= None,
            width= None,
            sigma_max= 4.0,
            sigma_min= 0.0,
        )

        
        z0 = np.zeros(14, dtype=float)
        x0 = 0.0
        h0 = 0.0
        y0 = 0.0
        self.state = np.concatenate([z0, [x0, h0, y0]])
        self.u = np.zeros(6, dtype=float)
        self.steps = 0

        # clear render traj
        self._traj_x = [x0]
        self._traj_h = [h0]
        self._traj_y = [y0]

        self.ayg_hist.clear()
        self.I_prev = 0.0
        self._just_reset = True


        obs = self.state.astype(np.float32)
        info = {}
        return obs, info
    
    def step(self, action: np.ndarray): 

        self.steps +=1

        action = np.asarray(action, dtype=float)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        val = action.copy()
        val = np.squeeze(val)
        self.u[0] = val[0] #aileron
        self.u[1] = val[1] #rudder
        self.u[5] = val[2] #elevator


        #gust input

        current_sigma = self._sigma_at_point(self.state[14], self.state[15])
        vg = current_sigma * self.rng.standard_normal()
        self.u[4] = vg


        boundary = 5
        B_turbed = self._create_B_turbed(self.B_base, sigma_val = current_sigma)
        next_state = self._rk4_step(state = self.state, B = B_turbed, u = self.u, dt = self.dt)
        next_state[15] = np.clip(next_state[15], self.observation_space.low[15] + boundary, self.observation_space.high[15] - boundary)
        next_state[16] = np.clip(next_state[16], self.observation_space.low[16] + boundary, self.observation_space.high[16] - boundary)
        ay_next = self._lateral_acceleration(z = next_state[:14], B = B_turbed, u_cmd = self.u) #check 
        

        self.ayg_hist.append(ay_next)
       


        I_next = float(np.sqrt(np.mean(np.square(self.ayg_hist)))) if self.ayg_hist else 0.0
        dI = 0.0 if self._just_reset else (I_next - self.I_prev)
        self._just_reset = False
        self.I_prev = I_next
        

        # dI = 0.0

        # if len(self.I_list) >= 2:
        #     dh = self.h_list[-1] - self.h_list[-2]
        #     dI = self.rmsaylist[-1] - self.rmsaylist[-2]
        #     # if abs(dh) > 1e-3: 
        #     #     dIdh = float(day / dh)
        #     #     dIdh = float(np.clip(dIdh, -5.0, 5.0))








        



        #-------------------------reward, termination and state update------------

        reward = (-abs(2*ay_next) - next_state[15] + next_state[14]) if self.reward_fn is None else float(self.reward_fn(next_state, self.state, dI, I_next, self.dt)) #we set our own reward function soon 

        terminated = ((next_state[14] >= 12000) or (next_state[15] > 201) or (next_state[15] < -201) or (next_state[16] > 201) or (next_state[16] < -201))
        truncated = (self.steps >= self.max_steps)
        self.state = next_state


        self._traj_x.append(self.state[14])
        self._traj_h.append(self.state[15])
        self._traj_y.append(self.state[16])

        info = {"ay": ay_next, "sigma": current_sigma}
        obs = self.state.astype(np.float32)

        if self.render_mode == "human": 
            self._render_human_frame()

        return obs, reward, terminated, truncated, info
    

        # ---------- rendering ----------
    def render(self):
        if self.render_mode == "human":
            self._render_human_frame()
        elif self.render_mode == "rgb_array":
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            return img
        else:
            return None

    def _render_human_frame(self):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 4.5))
            im = self._ax.imshow(
                self.sigma, origin="lower",
                extent=(self.X.min(), self.X.max(), self.Z.min(), self.Z.max()),
                aspect="auto", vmin = 0.0, vmax = 4.0
            )
            self._fig.colorbar(im, ax=self._ax, label="sigma (m/s)")
            self._ax.set_xlabel("Longitudinal distance (m)")
            self._ax.set_ylabel("altitude change (m)")
            self._ax.set_title("Trajectory over sigma field")
            self._ax.set_xlim(0,12000)
        # update trajectory line
        self._ax.plot(self._traj_x, self._traj_h, "k-", lw=2)
        plt.pause(0.001)

    def close(self):
        if self.render_mode == 'human' and self._fig is not None: 
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

            return

        if self._fig is not None:
            # plt.savefig('.\OneDrive\Year 5\Bio-inspired\data\ final image.png', dpi = 150)
            plt.close(self._fig)
            self._fig = None
            self._ax = None


    # ----------------- convenient registration helper -----------------
def register_citation_env(env_id: str = "Citation-nlgauss", **kwargs):

    if env_id in registry: 
        del registry[env_id]
    try:
        register(
            id=env_id,
            entry_point="citation_env_gauss:CitationEnvGauss",
            kwargs=kwargs,
            max_episode_steps= None),
    except Exception:
        
        pass