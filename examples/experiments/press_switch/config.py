import glob
import os
import jax
import jax.numpy as jnp
import numpy as np

from airbot_env.envs.wrappers import (
    Quat2EulerWrapper,
    # KeyboardIntervention,
    ArmIntervention,
    GripperPenaltyWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
    VipRewardClassifierWrapper
)
from airbot_env.envs.relative_env import RelativeFrame
from airbot_env.envs.airbot_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.open_drawer.wrapper import DrawerEnv

class EnvConfig(DefaultEnvConfig):
    ROBOT_IP = "127.0.0.1"
    ROBOT_PORT = 50051
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "243222072703",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "env_close": {
            "serial_number": "243322073753",
            "dim": (1280, 720),
            "exposure": 40000,
        },       
        # "env_far": {
        #     "serial_number": "243322072684",
        #     "dim": (1280, 720),
        #     "exposure": 40000,
        # }
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[160:720, 520:1100],
        "env_close": lambda img: img[100:660, 80:630],
        # "env_far": lambda img: img[0:500, 450:750],
    }
    TARGET_POSE = np.array([0.3866710066795349, -1.2563334703445435, 0.9621115326881409, 1.1063177585601807, -0.6192041039466858, -1.0273687839508057])
    RESET_POSE = [-0.03143169358372688, -0.4260631501674652, 0.8950029015541077, 1.6541465520858765, -1.1428875923156738, -1.7735157012939453]
    ABS_POSE_LIMIT_LOW = RESET_POSE - np.array([0.0, 0.10, 0.1, 0.2, 0.2, 0.2])
    ABS_POSE_LIMIT_HIGH = RESET_POSE + np.array([0.25, 0.10, 0.05, 0.2, 0.2, 0.2])

    SAFEBOX_MAX = [0.416, 0.095, 0.29, 0.183, 0.45, 0.07, 0.96]
    SAFEBOX_MIN = [0.191,-0.175,0.1,-0.26,0.27,-0.18,0.88]

    GRIPPER_IS_STATIC = True
    GRIPPER_STATIC_POS = 0
    GO_TO_RESET = True
    RANDOM_RESET = False
    RANDOM_RESET_MAX = [0] * 7
    RANDOM_RESET_MIN = [0] * 7
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.02
    ACTION_SCALE = (0.03, 0.03, 1)
    MAX_EPISODE_LENGTH = 250


class TrainConfig(DefaultTrainingConfig):
    # image_keys = ["wrist", "env_close", "env_far"]
    image_keys = ["env_close", "wrist"]
    # classifier_keys = ["wrist", "env_close", "env_far"]
    reward_keys = ["env_close_full"]
    classifier_keys = ["wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = DrawerEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        if not fake_env:
            # env = VRPicoItervation(env)
            # env = SpacemouseIntervention(env)
            # env = KeyboardIntervention(env)
            env = ArmIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            from PIL import Image
            import os
            
            # step reward classifier
            goal_imgs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_images")
            if not os.path.exists(goal_imgs_dir):
                os.makedirs(goal_imgs_dir)
                print(f"警告: 目标图片目录不存在，已创建空目录 {goal_imgs_dir}")
                print(f"请在该目录放入目标状态的图片后再运行")
                
            # 读取所有.png和.jpg图片
            image_paths = glob.glob(os.path.join(goal_imgs_dir, "*.png")) + \
                          glob.glob(os.path.join(goal_imgs_dir, "*.jpg"))
            
            if not image_paths:
                raise ValueError(f"目标图片目录 {goal_imgs_dir} 中没有找到图片文件")
            
            goal_imgs = []
            for img_path in image_paths:
                try:
                    # 打开并转换为RGB格式
                    img = Image.open(img_path).convert("RGB")
                    print(f"已加载目标图片: {img_path}")
                    goal_imgs.append(img)
                except Exception as e:
                    print(f"无法加载图片 {img_path}: {e}")
            
            print(f"共加载了 {len(goal_imgs)} 张目标图片")
            
            # 确认至少有一张目标图片
            if not goal_imgs:
                raise ValueError("没有成功加载任何目标图片")
                
            env = VipRewardClassifierWrapper(env, self.reward_keys, goal_imgs, score_factor=0.5)
            
            # success reward classifier
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path="/home/kkk/workspace/hil-serl/examples/experiments/press_switch/classifier_ckpt",
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                score = sigmoid(classifier(obs))
                print("success reliability score: ", score)
                return int(score.item() > 0.80)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        env = GripperPenaltyWrapper(env, penalty=-0.1)

        return env