# gym.envs.mjrl envs
from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly


from gym.envs.mjrl.dkitty_button import ButtonEnv
from gym.envs.mjrl.dkitty_switch import SwitchEnv
from gym.envs.mjrl.dkitty_button_random import ButtonRandomEnv
from gym.envs.mjrl.dkitty_switch_random import SwitchRandomEnv