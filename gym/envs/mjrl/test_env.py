import gym
import numpy as np
import time

# import gym.envs.robotics.mjrl.envs
from mjrl.utils.gym_env import GymEnv

mode = 'human'
#mode = 'rgb_array'

# env = gym.make("RexROVURReach-v1")
# env = gym.make("RexROVURPipeMonitor-v1")
# env = gym.make("mjrl_rexrov2_ur3_pick-v0")
# env = GymEnv("mjrl_rexrov2_ur3_pick-v0")
env = GymEnv("mjrl_dkitty_button-v0")
# env = GymEnv("mjrl_rexrov2_ur3_insertion-v0")
# env = GymEnv("")
# env = gym.make("mjrl_floating_peg_insertion-v0")

# print("action space high: ", env.action_space.high)
# print("action space low: ", env.action_space.low)
# num_actuator = env.sim.model.nu
# print('num_actuator: ', num_actuator)
# time.sleep(10)
# print('joint names:', env.sim.model.joint_names)
# print("robot_base_link: ", env.sim.data.get_body_xpos("robot0:base_link"))
# print("r_ur5_arm_wrist_3_link: ", env.sim.data.get_body_xpos("r_ur5_arm_wrist_3_link"))
# print("ee_link pos w.r.t base_link: ", env.sim.data.get_body_xpos('ee_link') - env.sim.data.get_body_xpos("robot0:base_link"))
# print("ee_link pos: ", env.sim.data.get_body_xpos('ee_link'))
# print("ee_link quat: ", env.sim.data.get_body_xquat('ee_link'))
# ee_link quat:  [-0.49959828  0.50040156 -0.50025185  0.49974786] w,x,y,z
# print("table0 pos: ", env.sim.data.get_body_xpos('table0'))
# print("object0 pos: ", env.sim.data.get_body_xpos('object0'))

# get the relative pos between two bodys for connect
# <connect body1="left_inner_knuckle" body2="left_inner_finger" anchor="0.05 -0.0 0.0"/>
# <connect body1="right_inner_knuckle" body2="right_inner_finger" anchor="0.05 -0.0 0.0"/>
# print("left_inner_finger: ", env.sim.data.get_body_xpos('left_inner_finger'))
# print("left_inner_knuckle: ", env.sim.data.get_body_xpos('left_inner_knuckle'))
# print("right_inner_finger: ", env.sim.data.get_body_xpos('right_inner_finger'))
# print("right_inner_knuckle: ", env.sim.data.get_body_xpos('right_inner_knuckle'))

# print("r_gripper_palm_link: ", env.sim.data.get_body_xpos('r_gripper_palm_link'))
# print("r_grip_site w.r.t base_link: ", env.sim.data.get_site_xpos("r_grip_site") - env.sim.data.get_body_xpos("robot0:base_link"), env.sim.data.get_site_xmat("r_grip_site"))
# print("r_grip_site: ", env.sim.data.get_site_xpos("r_grip_site"), env.sim.data.get_site_xmat("r_grip_site"))
# print("dist ee w.r.t grip: ", (env.sim.data.get_body_xpos('ee_link')-env.sim.data.get_site_xpos('r_grip_site')))
# print("ee_link: ", env.sim.data.get_body_xpos('ee_link'))
# print("ee_link w.r.t base_link: ", env.sim.data.get_body_xpos('ee_link')-env.sim.data.get_body_xpos("robot0:base_link"))
# print("ee_link pos: ", env.sim.data.get_body_xpos('ee_link'))
# print("ee_link xquat: ", env.sim.data.get_body_xquat('ee_link'))

# print("qpos: ", env.sim.data.qpos)
# print("ctrl: ", env.sim.data.ctrl)
# print("action_space: ", env.action_space)
# print("observation space high: ", env.observation_space.high)
# print("observation space low: ", env.observation_space.low)
# print("observation_space", env.observation_space)
# print("sim.model.actuator_biastype: ", env.sim.model.actuator_biastype)
# print("sim.model.nmocap: ", env.sim.model.nmocap)
# print("sim.model.mocap_quat: ", env.sim.data.mocap_quat[0,:])
# print("sim.model.mocap_pos: ", env.sim.data.mocap_pos[0,:])
# print("ur5 joint1: ", env.sim.data.get_joint_qpos("r_ur5_arm_shoulder_pan_joint"))
# print("ur5 joint2: ", env.sim.data.get_joint_qpos("r_ur5_arm_shoulder_lift_joint"))
# print("ur5 joint3: ", env.sim.data.get_joint_qpos("r_ur5_arm_elbow_joint"))
# print("ur5 joint4: ", env.sim.data.get_joint_qpos("r_ur5_arm_wrist_1_joint"))
# print("ur5 joint5: ", env.sim.data.get_joint_qpos("r_ur5_arm_wrist_2_joint"))
# print("ur5 joint6: ", env.sim.data.get_joint_qpos("r_ur5_arm_wrist_3_joint"))
# print("jnt_qposadr: ", env.sim.model.jnt_qposadr)
# print("sim.model.actuator_trnid: ", env.sim.model.actuator_trnid)

# for i in range(11):
#       print("\n num: ", i)
#       print("sim.model.actuator_trnid: ", env.sim.model.actuator_trnid[i,0])
#       print("sim.model.jnt_qposadr: ", env.sim.model.jnt_qposadr[env.sim.model.actuator_trnid[i, 0]])
# idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
# env.render('human')
#env = gym.wrappers.Monitor(env, './video', force=True)
# plt.imshow(env.render(mode='rgb_array', camera_id=0))
# plt.show()
# time.sleep(10)
# plt.show()
# action = np.array([0.1, 0.0, 0.0, -0.1])
for j in range(10):
  env.reset()
  # env.render('human')
  # action[-1] += 0.1
  # if action[-1] > 0.01:
        # action[-1] = -1.0
  # action = np.array([0., 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 1.0])
  for i in range(10):
      action = env.action_space.sample()
      # if i % 10 == 0:
        # action[-1] = 1
    #   action = np.array([0., 0., 0., 0., 0., 0., -0., 0.0, -0., 0., 0., 0., -1])
      # for k in range(13):
      #   action[k] = 1.0
      #   env.step(action)
      #   time.sleep(1)
      #   env.render('human')
      # action = np.array([0., -0., 0., 0., 0., -0., -1]) # 1.0 open -1.0 close
      # action = np.array([0.0, 0.0, 0.0, 0.0, -1.0]) # 1.0 open -1.0 close
      # action[-1] += 0.1
      # if action[-1] > 1.0:
            # action[-1] = -1.0
      # env.sim.data.set_mocap_pos('table0:mocap', [0.5,0.8,1.0])
      # env.sim.data.set_mocap_quat('table0:mocap', [1.,0,0,0])
      # env.sim.data.mocap_pos[1,:] = [0.5,-0.8,1.0] 
      # env.sim.data.mocap_quat[1,:] = [1,0,0,0]     
      print("action_space:", env.env.action_space)
      print("action: ", action)
      # action[0] = 0
      # env.env.data.qpos[6] = -3.14
      obs, reward, done, info = env.env.step(action)
      # print("torso: ", env.env.data.get_body_xpos("A:kitty_frame"))
      # print("button_up: ", env.env.data.get_site_xpos("button_up"))
      # print("button_down: ", env.env.data.get_site_xpos("button_down"))
      print("qpos: ", env.env.data.qpos)
      print("init_qpos: ", env.env.init_qpos)
      print("observation:", obs)
      print("reward:", reward)
      print("done:", done)
      print("info:", info)
      # print("init_body_pos: ", env.env.init_body_pos)
    #   print("eef_force_sensor_idx: ", env.eef_force_sensor_idx)
    #   print("eef_torque_sensor_idx: ", env.eef_torque_sensor_idx)
      # print("sensordata: ", env.env.sim.data.sensordata)
      # print("torso linear vel: ", env.env.torso_linear_vel)
      # print("torso angular vel: ", env.env.torso_angular_vel)      
      # print("force: ", env.env.ee_force)
      # print("torque: ", env.env.ee_torque)

      env.render()