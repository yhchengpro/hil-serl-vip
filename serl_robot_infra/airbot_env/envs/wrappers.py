import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R
from typing import List
from airbot_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from airbot_env.keyboard.keyboard_expert import KeyboardExpert
from airbot_env.utils.transformations import (
    construct_homogeneous_matrix,
)

import argparse
import time
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
from pynput import keyboard
from .airbot_kdl import ArmKdl
from airbot_py.arm import AIRBOTArm
import threading
import math
from typing import Sequence,Tuple

import tkinter as tk
import time

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class VipRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, image_key: List[str], goal_imgs: List, score_factor: float = 1.0, smooth_weight: float = 0.05):
        super().__init__(env)
        from vip import load_vip
        from torchvision import transforms as T
        
        self.model = load_vip()
        self.model.to("cuda")
        self.model.eval()

        self.transform = T.Compose([T.Resize(224),
                # T.CenterCrop(224),
                T.ToTensor()])
        self.goal_imgs = goal_imgs
        
        self.last_distance = None
        self.last_action = None
        
        
        self.image_key = image_key
        self.score_factor = score_factor
        self.smooth_weight = smooth_weight

    def compute_reward(self, obs):
        time1 = time.time()
        for key in self.image_key:
            if key not in obs:
                raise ValueError(f"Image key {key} not found in observation")
            img = Image.fromarray(obs[key][0].astype(np.uint8))
            imgs = [*self.goal_imgs, img]
            image = torch.stack([self.transform(img) for img in imgs])
        print(f"Image processing took {time.time() - time1:.4f}s")
            
        with torch.no_grad():
            time1 = time.time()
            embeddings = self.model(image)
            print(f"VIP inference took {time.time() - time1:.4f}s")
            
        distance = (embeddings[-1] - embeddings[:-1]).norm(dim=1).min().item()
        
        if self.last_distance == None:
            rew = 0
            
        elif distance < 0.01:
            rew = 5
        else:
            alpha = 0.1
            distance = alpha * distance + (1.0 - alpha) * self.last_distance
            print("\033[35m {}\033[00m".format(f"distance = {distance}"))
            rew = np.round((self.last_distance - distance), 2) * self.score_factor
            
        self.last_distance = distance
        print(f"Distance to goal: {distance:.4f}, reward: {rew:.4f}")
        
        return rew

    def step(self, action):
        time1 = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew += self.compute_reward(obs) - 0.01 # penalty for each step
        print(f"vip reward took {time.time() - time1:.4f}s")
        
        done = done or rew == 5
        info['succeed'] = bool(rew == 5)

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        self.last_distance = None
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        done = done
        info['succeed'] = False
        time1 = time.time()
        if self.compute_reward(obs):
            print("\033[92m {}\033[00m".format("SUCCESS RESET"))
            rew += 5
            done = True
            info['succeed'] = True
        print(f"Reward computation took {time.time() - time1:.4f}s")
            
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class KeyboardIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        
        self.expert = KeyboardExpert()
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, expert_e, intervened = self.expert.get_action()

        if intervened:
            if self.gripper_enabled:
                expert_a = np.concatenate((expert_a, expert_e), axis=0)
            return expert_a, True
        
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        # info["left"] = self.left
        # info["right"] = self.right
        return obs, rew, done, truncated, info
    
class ArmIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        
        self.intervened = False
        self._start_listener()
        self.arm_kdl = ArmKdl(eef_type="G2")
        self.lock = threading.Lock()
        self.robot1 = AIRBOTArm("localhost", 50050)
        self.robot1.connect()
        self.last_end_pos = self.robot1.get_end_pose()
        self.end_pos = self.last_end_pos
        self.last_pos = self.endpose_2_pos(self.last_end_pos)
        self.pos = self.endpose_2_pos(self.end_pos)
        time.sleep(0.2)
        print("Lead robot arm is ready to follow.")
    
    def _start_listener(self):
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.daemon = True
        self.listener.start()
    
    def _on_press(self, key):
        try:
            with self.lock:
                if key == keyboard.Key.enter:
                    self.intervened = not self.intervened   
                    print("intervened")
        except Exception as e:
            print(f"_on_press error: {e}")
    
    def endpose_2_pos(self, end_pose):
        pos = end_pose[0]
        
        euler = R.from_quat(end_pose[1]).as_euler('xyz')
        return np.concatenate([pos, euler], axis=0)

    def action(self, action: np.ndarray) -> np.ndarray:
        self.end_pos = self.robot1.get_end_pose()    
        self.pos = self.endpose_2_pos(self.end_pos)
        print(f"get_end_pos{self.end_pos}")
        # print(f"pos{self.pos}")
        self.diff_action = (self.pos - self.last_pos)
        self.diff_action[:3] *= 15
        self.diff_action[3:] *= 10
        
        print(f"diff_action{self.diff_action}")
        self.gripper_pos = np.array(self.robot1.get_eef_pos())
        print(f"gripper_pos{self.gripper_pos}")
        self.last_pos = self.pos
        
        if self.intervened:
            follow_action = np.concatenate((self.diff_action, self.gripper_pos), axis=0)
            # print(f"follow_action{follow_action}")
            return follow_action, True
        return action, False
    
    def step(self, action):

        new_action, replaced = self.action(action)
        if self.intervened:
            print("\033[93m {}\033[00m".format(f"new_action{new_action}"))
        else:
            print(f"new_action{new_action}")

        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
    
        if hasattr(base_env, 'set_intervention_state'):
            base_env.set_intervention_state(replaced)
  
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
            
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True
        
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=-0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
            
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < -0.95
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info

class VRPicoItervation(gym.ActionWrapper):
    """
    Use this wrapper to intervene with the VR Quest controller.
    It will use the right controller to control the robot.
    """

    def __init__(self, env, action_indices=None):
        from airbot_data_collection.airbot.teleoprators.vr_pico import VRPicoController, TeleopConfig, EventConfig, VREvent
        
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False
            
        self.expert = VRPicoController(
            TeleopConfig(
                event_config=EventConfig(
                    zero_info=VREvent.RIGHT_GRIP,
                    success=VREvent.RIGHT_STICK_H,
                    failure=VREvent.RIGHT_STICK_H,
                    rerecord_episode=VREvent.RIGHT_PRIMARY_BUTTON,
                    intervention=VREvent.RIGHT_GRIP,
                    shutdown=VREvent.RIGHT_SECONDARY_BUTTON,
                    left_eef=VREvent.LEFT_STICK_V,
                    right_eef=VREvent.RIGHT_STICK_V,
                ),
            )
        )
        self.expert.start()
        self.left, self.right = False, False
        self.action_indices = action_indices
        self.intervened = False
        self.pose = None
        self.T = None
        self.T_inv = None
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        self.expert.update()
        intervened  = self.expert.should_intervene()
        expert_a = np.array(self.expert.get_deltas("right"))
        
        if not intervened or (np.linalg.norm(expert_a) < 0.001 and self.expert.gripper_command() == "stay"):
            self.intervened = intervened
            pose = self.env.currpos
            trans = action[:3] + pose[:3]
            orient = (R.from_quat(pose[3:]) *R.from_euler("xyz", action[3:6])).as_quat()
            action = [*trans,*orient, action[6]]
            return action, False
        else:
            if not self.intervened:
                self.pose = self.env.currpos
                # self.T = construct_homogeneous_matrix(self.pose)
                # self.T_inv = np.linalg.inv(self.T)

            # self.intervened = intervened
            
            # if np.linalg.norm(expert_a) > 0.001:
            #     expert_a = self.T_inv @ construct_homogeneous_matrix(expert_a) @ self.T
            # elif self.expert.gripper_command() is not "stay":
            #     expert_a = [0,0,0,0,0,0,1]
                
            # trans = expert_a[:3,3] + self.pose[:3]
            # orient = (R.from_matrix(expert_a[:3,:3])* R.from_quat(self.pose[3:])).as_quat()
            # expert_a = [*trans,*orient]
            if np.linalg.norm(expert_a) < 0.001 and self.expert.gripper_command() == "stay":
                expert_a = [0,0,0,0,0,0,1]
                
            trans = self.config.ACTION_SCALE[0] * expert_a[:3] + self.pose[:3]
            orient_euler = self.config.ACTION_SCALE[1] * (R.from_quat(expert_a[3:]).as_euler("xyz"))
            print(f"orient_euler: {orient_euler}")
            orient = (R.from_quat(self.pose[3:])* R.from_euler("xyz",orient_euler)).as_quat()
            expert_a = [*trans,*orient]
        
            self.expert._vr._tf_pub.broadcast_tf(
            position = self.pose[:3],
            orientation = self.pose[3:],
            child_frame_id = "tool", 
            )

            self.expert._vr._tf_pub.broadcast_tf(
            position = trans,
            orientation = orient,
            child_frame_id = "tool_rela", 
            )

            expert_eef = self.expert.gripper_command()
            gripper = self.env.curr_gripper_pos
            if expert_eef == "open":
                gripper = [1]
            elif expert_eef == "close":
                gripper = [0]
            
            expert_a = np.concatenate((expert_a, gripper), axis=0)
            
            # print(f"intervene: {intervened}, action: {np.round(expert_a,4)}, gripper: {gripper}")
            return expert_a, True

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info