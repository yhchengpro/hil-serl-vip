"""Gym Interface for Airbot"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Callable

from airbot_env.camera.video_capture import VideoCapture
from airbot_env.camera.rs_capture import RSCapture
from airbot_env.utils.rotations import euler_2_quat, quat_2_euler

from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile
from .airbot_kdl import ArmKdl
from airbot_env.utils.transformations import construct_homogeneous_matrix

from airbot_env.utils.lpf import MultiDimLPF
import random
from scipy.spatial.transform import Rotation as R

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (256, 256)) for k, v in img_array.items()], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)

# class ImageDisplayer(threading.Thread):
#     def __init__(self, img_queue, win_prefix="cam"):
#         super().__init__(daemon=True)
#         self.img_queue = img_queue
#         self.win_prefix = win_prefix
#         self._running = True

#     def run(self):
#         import cv2, time
#         while self._running:
#             try:
#                 item = self.img_queue.get(timeout=0.2)
#             except queue.Empty:
#                 continue
#             if item is None:
#                 break
#             # item: dict of images -> display
#             for k, im in item.items():
#                 cv2.imshow(f"{self.win_prefix}_{k}", im[..., ::-1])  # BGR->RGB if needed
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass


##############################################################################


class DefaultEnvConfig:
    """Default configuration for AirbotEnv. Fill in the values below."""

    ROBOT_IP = "127.0.0.1"
    ROBOT_PORT = 50051
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: Dict[str, Callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.5
    MAX_EPISODE_LENGTH: int = 10000
    JOINT_RESET_PERIOD: int = 0
    GRIPPER_WIDTH: float = 0.072 # m


##############################################################################


class AirbotEnv(gym.Env):
    def __init__(
        self,
        hz=50,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
    ):
        self.lpf = MultiDimLPF(dims=6, fc=0.5, kp=200, kd=0.2, v_max=0.1, a_max=0.1)
        self.last_time = time.time()
        self.cnt = 0
        self.control_hz = 250
        self._state_lock = threading.Lock()
        
        self.robot = AIRBOTArm(config.ROBOT_IP, config.ROBOT_PORT)
        self.robot.connect()
        self.robot.set_speed_profile(SpeedProfile.DEFAULT)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)   
        # eef = self.robot.get_product_info()["eef_types"][0]
        self.arm_kdl = ArmKdl(eef_type="G2")
        self.mode = RobotMode.SERVO_JOINT_POS
        
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.fake_env = fake_env

        # Initialize robot state variables
        self.currpose = np.zeros(7)  # [x, y, z, qx, qy, qz, qw]
        self.currvel = np.zeros(6)   # [vx, vy, vz, wx, wy, wz]
        self.curr_gripper_pos = np.array([1.0])  # gripper position
        self.currforce = np.zeros(3)
        self.currtorque = np.zeros(3)
        self.curr_path_length = 0

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self.last_gripper_pos = 1.0
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        self._obs_lock = threading.Lock()
        self.obs = None
        self.img_queue = None
        self.displayer = None
        
        self.last_time = time.time()
        self.cnt = 0
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8) 
                                for key in config.REALSENSE_CAMERAS}
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, config.ROBOT_IP)
            self.displayer.start()


        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        
        self._is_intervention = False
        
        self._running = True
        self._state_thread = threading.Thread(target=self._state_update_loop, daemon=True)
        self._state_thread.start()
        self.arr=[]
        self.safebox_max = config.SAFEBOX_MAX
        self.safebox_min = config.SAFEBOX_MIN
        self.gripper_is_static = config.GRIPPER_IS_STATIC
        if self.gripper_is_static:
            self.gripper_static_pose = config.GRIPPER_STATIC_POS
        if self.randomreset:
            self.random_reset_max = config.RANDOM_RESET_MAX
            self.random_reset_min = config.RANDOM_RESET_MIN  
        self._go_to_reset = config.GO_TO_RESET
        time.sleep(1)

        print("Initialized Airbot")
    
    def _state_update_loop(self):
        """以 10ms 的频率更新一次状态"""
        while self._running:
            try:
                self._update_obs()
            except Exception as e:
                print(f"[WARN] Failed to update robot state: {e}, trying to reconnect")
                self.robot.connect()
                time.sleep(1)
            time.sleep(0.01)

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        for i in range(6):
            pose[i] = np.clip(pose[i], self.safebox_min[i], self.safebox_max[i])
        return pose

    def set_intervention_state(self, is_intervention: bool):
        self._is_intervention = is_intervention

    def step(self, action: np.ndarray, status: str = "") -> tuple:
        now = time.time()
        if self.gripper_is_static:
            action[6] = self.gripper_static_pose

        is_intervention = getattr(self, "_is_intervention", False)
        self._is_intervention = False

        if not is_intervention:
            # 先把原始 action 按 config 缩放
            action[:3] = self.config.ACTION_SCALE[0] * action[:3]
            action[3:6] = self.config.ACTION_SCALE[1] * action[3:6]
            self.lpf.update(now, action[:6])

            # 在当前时间采样平滑后的结果
            smoothed_action = np.array(self.lpf.sample(now))
        else:
            smoothed_action = action[:6].copy()
        print("\033[94m {}\033[00m".format(f"smoothed_command{smoothed_action}"))

        # 用平滑后的 action 控制
        self._send_gripper_command(float(action[6]))
        print(f"Raw: {np.round(action,3)}, Smoothed: {np.round(smoothed_action,3)}")

        if np.linalg.norm(smoothed_action) > 0.001:
            self._send_pos_command(smoothed_action)

        self.curr_path_length += 1
        dt = time.time() - now
        time.sleep(max(0, (1.0 / self.hz) - dt))

        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length #or reward or self.terminate
        if self.curr_path_length >= self.max_episode_length:
            print("\033[91m {}\033[00m".format("TIMEOUT RESET"))
        self.cnt += 1
        if (time.time() - self.last_time) > 1.0:
            print(f"\n --------FPS: {self.cnt}--------- \n")
            self.cnt = 0
            self.last_time = time.time()
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        # current_pose = obs["state"]["tcp_pose"]
        # # convert from quat to euler first
        # current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        # target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        # diff_rot = current_rot.T  @ target_rot
        # diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        # delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # # print(f"Delta: {delta}")
        # if np.all(delta < self._REWARD_THRESHOLD):
        #     return True
        # else:
        #     # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
        #     return False
        return 0

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                resized_rgb = cv2.resize(
                    rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized_crop = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized_crop[..., ::-1]
                if key == "env_close":
                    images[key + "_full"] = resized_rgb[..., ::-1]
                display_images[key] = resized_crop
                display_images[key + "_full"] = resized_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image and self.img_queue is not None:
            self.img_queue.put(display_images)
        return images

    def go_to_reset(self):
        """_send_gripper_command
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Perform Carteasian reset
        if self.mode != RobotMode.PLANNING_POS:
            self.robot.switch_mode(RobotMode.PLANNING_POS)
            self.mode = RobotMode.PLANNING_POS
            
        if self.randomreset:  # randomize reset position in xy plane
            random.seed(time.time())
            random_pose = [random.uniform(mn, mx) for mn, mx in zip(self.random_reset_min, self.random_reset_max)]
            pos = [random_pose[:3], random_pose[3:]]
            self.robot.move_to_cart_pose(pos)
        else:
            reset_pose = self._RESET_POSE        
            time.sleep(0.5)
            self.robot.servo_joint_pos(reset_pose)
        time.sleep(0.4)
        
        if self.gripper_is_static:
            self.curr_gripper_pos = self.gripper_static_pose
            self.robot.move_eef_pos(self.gripper_static_pose)
            time.sleep(0.4)
        self.lpf.update(time.time(), np.zeros((6,)))
        self.lpf.reset()
        self.lpf = MultiDimLPF(dims=6, fc=2.0, v_max=0.1, a_max=0.5)

        print("\n\nReset done\n\n")
    
    def pull_back(self):
        if self.mode != RobotMode.PLANNING_POS:
            self.robot.switch_mode(RobotMode.PLANNING_POS)
            self.mode = RobotMode.PLANNING_POS
        
        self.curr_gripper_pos = -1
        self.robot.move_eef_pos(0)
        time.sleep(0.5)
        curpos = self.robot.get_end_pose()
        curpos[0][0] -= 0.15
        self.robot.move_to_cart_pose(curpos)

    def reset(self, **kwargs):
        print("\033[35m {}\033[00m".format("----------------RESET!!!----------------"))
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # if self._go_to_reset:
        #     self.go_to_reset()
        self.go_to_reset()
        self.curr_path_length = 0

        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            if self.cap is not None:
                for cap in self.cap.values():
                    cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _send_pos_command(self, action: np.ndarray):
        """Internal function to send position command to the robot."""
        action[3]=0
        action[4]=0
        action[5]=0
        if self.mode != RobotMode.SERVO_JOINT_POS:
            self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)    
            self.mode = RobotMode.SERVO_JOINT_POS
        curpose = self.robot.get_end_pose()
        curjoint = self.robot.get_joint_pos()
        trans = action[:3] + curpose[0]
        #orient = (Rotation.from_quat(curpose[1]) * Rotation.from_euler("xyz", self.config.ACTION_SCALE[1] * action[3:6])).as_quat()
        orient = (Rotation.from_quat(curpose[1]) * Rotation.from_euler("xyz", action[3:6])).as_quat()
        pose = list([*trans, *orient])
        eular = R.from_quat(orient).as_euler('xyz')
        pose_6 = list([*trans, *eular])
        self.arr.append(pose)
        arr_np = np.array(self.arr)

        # 计算每个维度（x, y, z）的最大值和最小值
        max_x = np.max(arr_np[:, 0])  # x 维度的最大值
        min_x = np.min(arr_np[:, 0])  # x 维度的最小值
        max_y = np.max(arr_np[:, 1])  # y 维度的最大值
        min_y = np.min(arr_np[:, 1])  # y 维度的最小值
        max_z = np.max(arr_np[:, 2])  # z 维度的最大值
        min_z = np.min(arr_np[:, 2])  # z 维度的最小值
        
        max_p = np.max(arr_np[:, 3])  # x 维度的最大值
        min_p = np.min(arr_np[:, 3])  # x 维度的最小值
        max_q = np.max(arr_np[:, 4])  # y 维度的最大值
        min_q = np.min(arr_np[:, 4])  # y 维度的最小值
        max_r = np.max(arr_np[:, 5])  # z 维度的最大值
        min_r = np.min(arr_np[:, 5])  # z 维度的最小值
        max_s = np.max(arr_np[:, 6])  # z 维度的最大值
        min_s = np.min(arr_np[:, 6])  # z 维度的最小值
        

        # 打印最大值和最小值
        print(f"Max values: [{max_x},{max_y},{max_z},{max_p},{max_q},{max_r},{max_s}]")
        print(f"Min values: [{min_x},{min_y},{min_z},{min_p},{min_q},{min_r},{min_s}]")
        safe_pose = self.clip_safety_box(pose)
        joints = self.arm_kdl.inverse_kinematics(construct_homogeneous_matrix(safe_pose), curjoint)
        if joints is None or len(joints) == 0:
            return
        target_joints = list(joints[0])
        self.robot.servo_joint_pos(target_joints)
        # print("Action: ", action, "Current pos:, ", curpose, "Sending pose: ", safe_pose, "Current joints:", curjoint, "Sending joints:", joints[0])
        
    def _send_vel_command(self, pose: np.ndarray):
        if self.mode != RobotMode.SERVO_CART_TWIST:
            self.robot.switch_mode(RobotMode.SERVO_CART_TWIST)    
            self.mode = RobotMode.SERVO_CART_TWIST  
        pose = list([[*pose[:3]], [*pose[3:]]])
        print("Sending cartesian twist command: ", pose)
        self.robot.servo_cart_twist(pose)

    def _send_gripper_command(self, pos: float, mode="continuous"):
        """Internal function to send gripper command to the robot."""
        # print("Sending gripper command: ", pos, "curr: ", self.curr_gripper_pos)
        if mode == "binary":
            if (pos <= -0.8) and (self.curr_gripper_pos > 0.8) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                self.robot.move_eef_pos(0)
                self.curr_gripper_pos = -1
                self.last_gripper_act = time.time()
                # time.sleep(self.gripper_sleep)
                return True
            elif (pos >= 0.8) and (self.curr_gripper_pos < -0.8) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                self.robot.move_eef_pos(1)
                self.curr_gripper_pos = 1
                self.last_gripper_act = time.time()
                # time.sleep(self.gripper_sleep)
                return True
            else: 
                return False
        elif mode == "continuous":
            pos = np.clip(pos, 0, 0.072)
            self.robot.move_eef_pos(pos)

    def _update_obs(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        self.q = np.array(self.robot.get_joint_pos())
        self.dq = np.array(self.robot.get_joint_vel())
        self.currjacobian = np.array(self.arm_kdl.jacobian(self.q))
        
        pose = self.robot.get_end_pose()
        
        self.currpos = np.array(pose[0] + pose[1]).tolist()
        self.currvel = np.array(self.currjacobian @ self.dq)
        self.curr_gripper_pos = np.array([1 if self.robot.get_eef_pos()[0] > 0.036 else -1])  
        
        tau = self.robot.get_joint_eff()
        f = np.linalg.pinv(self.currjacobian.T) @ tau

        self.currforce = np.array(f[:3])
        self.currtorque = np.array(f[3:])
        
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        with self._obs_lock:
            self.obs = copy.deepcopy(dict(images=images, state=state_observation))

    def _get_obs(self) -> dict:
        while self.obs is None:
            time.sleep(0.01)
        with self._obs_lock:
            return copy.deepcopy(self.obs)

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.robot.disconnect()
        if self.fake_env:
            return

        self._running = False
        if hasattr(self, '_state_thread'):
            self._state_thread.join()
        if hasattr(self, '_obs_thread'):
            self._obs_thread.join()
        if hasattr(self, '_control_thread'):
            self._control_thread.join()
            
        self.close_cameras()
        if self.display_image and self.img_queue is not None:
            self.img_queue.put(None)
        if self.displayer is not None:
            self.img_queue.put(None)
            self.displayer.join(timeout=1) 
        cv2.destroyAllWindows()
    