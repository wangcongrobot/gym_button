import numpy as np
from gym import utils
from gym.envs.mjrl import mujoco_env
from mujoco_py import MjViewer
import random

class SwitchEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.button_on = True
        self.peg_sid = -2
        self.target_sid = -1
        self.button_on_init_pos = 0.40
        self.button_center_pos = 0.395
        self.button_off_init_pos = 0.39
        self.button_state = 0 # 0: on, 1: off
        self.timestep = 0
        self._is_success = False
        self.last_action = None
        self.last_state = None
        mujoco_env.MujocoEnv.__init__(self, 'dkitty/switch.xml', 1)
        utils.EzPickle.__init__(self)
        self.peg_sid = self.model.site_name2id("A:FLfoot")
        self.target_sid = self.model.site_name2id("button_tip")
        self.init_body_pos = self.model.body_pos.copy()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()
        reward = self.get_reward(obs, a)
        self.timestep += 1
        print("done: ", self._is_success)
        done = False
        info = {
            'is_success': self._is_success,
        }
        self.last_action = a
        return obs, reward, done, info

    def get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            self.sim.data.get_site_xpos('A:FLfoot'),
            self.sim.data.get_site_xpos('A:FLfoot') - self.sim.data.get_site_xpos('button_tip'),
            self.sim.data.get_site_xpos('button_tip'),
            [self.button_state],
            [self._is_success],
        ])

    def get_reward(self, obs, act=None):
        r_reach = 0
        r_bonus = 0
        r_time_penalty = 0
        action_jerky_penalty = 0
        opposite_penalty = 0
        r_total = 0
        if self.last_action is not None:
            # print("last_action: ", self.last_action)
            # print("current_action: ", act)
            error = act - self.last_action
            if max(np.abs(error)) > 0.1:
                action_jerky_penalty = -0.1 * max(np.abs(error))
            # print("error: ", error)
            # print("action_jerky_penalty: ", action_jerky_penalty)
        
        hand = self.sim.data.get_site_xpos('A:FLfoot')
        button_tip_site = self.sim.data.get_site_xpos('button_tip')

        r_time_penalty = -0.0 * self.timestep
        # print("button_tip_site: ", button_tip_site)
        # print("hand: ", hand)

        if self.button_state == 0: # on, press the button_tip: [0.4        0.34530444 0.36511974]
            print("button on")
            r_reach = - np.linalg.norm(hand - button_tip_site) # + np.exp(-np.linalg.norm(hand - button_tip_site) ** 2 / 0.01)
            # print("dist: ", np.linalg.norm(hand - button_tip_site))
            if np.linalg.norm(hand - button_tip_site) < 0.05: # 0.374 turn to off
                r_reach += 0.1
                if button_tip_site[2] < 0.32: # button off
                    r_bonus = 5.0
                    self._is_success = True
        if self.button_state == 1: # off, press the button_tip: [0.4        0.34089567 0.30295045]
            print("button off")
            r_reach = - np.linalg.norm(hand - button_tip_site) # + np.exp(-np.linalg.norm(hand - button_tip_site) ** 2 / 0.01)
            # print("dist: ", np.linalg.norm(hand - button_tip_site))     
            if np.linalg.norm(hand - button_tip_site) < 0.05: # 0.397 turn to on
                r_reach += 0.1
                if button_tip_site[2] > 0.35: # button on
                    self._is_success = True
                    r_bonus = 5.0
        r_total = r_reach + r_bonus + r_time_penalty + action_jerky_penalty + opposite_penalty
                   
        # print("r_reach: ", r_reach)
        # print("r_bonus: ", r_bonus)
        # print("r_time_penalty: ", r_time_penalty)  
        # print("opposite_penalty: ", opposite_penalty)  
        # print("r_total: ", r_total)                
        return r_total

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        # choose button initial state randomly
        try:
            self.data.qpos[-1] = random.choice([1.,2.])
            # print("qpos0: ", self.data.qpos)
            if self.data.qpos[-1] == 1.: # on
                self.button_state = 0
            if self.data.qpos[-1] == 2.: # off
                self.button_state = 1
            # print("qpos1: ", self.data.qpos)
            self.data.qpos[0] = np.random.uniform(-1.0, 1.0)
            self.data.qpos[1] = np.random.uniform(-1.0, 1.0)
            self.data.qpos[2] = np.random.uniform(-1.0, 1.0)
            for _ in range(20):
                self.sim.step()
            # print("qpos2: ", self.data.qpos)
        except:
            pass

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        # print("reset_model")
        self.target_reset()
        self.timestep = 0
        self._is_success = False
        self.last_action = None
        return self.get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.body_pos[-1].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        print("qp, qv ", qp, qv)
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.sim.forward()
        # print("set_env_state")

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth += 200
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent*2.0
