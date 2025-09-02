import copy
import time
from airbot_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from pynput import keyboard

from airbot_env.envs.airbot_env import AirbotEnv

class DrawerEnv(AirbotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()