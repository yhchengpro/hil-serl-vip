import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
import time
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful groups to collect.")


reset_key = False
save_and_reset_key = False

def on_press(key):
    global reset_key, save_and_reset_key
    try:
        if hasattr(key, 'char') and key.char in ['r', 'R']:
            reset_key = True
            print("Reset Key Pressed! (不保存)")
        elif key == keyboard.Key.space:
            save_and_reset_key = True
            print("Space Key Pressed! (保存并复位)")
    except AttributeError:
        pass

def main(_):
    global reset_key, save_and_reset_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    groups = []
    
    # 使用列表分别存储每个episode的数据
    observations_list = []
    actions_list = []
    next_observations_list = []
    masks_list = []
    dones_list = []
    
    group_idx = 0
    groups_needed = FLAGS.successes_needed
    pbar = tqdm(total=groups_needed)

    while len(groups) < groups_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # 收集数据到列表
        observations_list.append(copy.deepcopy(obs))
        actions_list.append(copy.deepcopy(actions))
        next_observations_list.append(copy.deepcopy(next_obs))
        masks_list.append(1.0 - done)
        dones_list.append(done)
        
        obs = next_obs

        # 检查按键状态
        if save_and_reset_key or reset_key or done or truncated:
            if len(observations_list) > 0 and (save_and_reset_key):
                # 计算准确的progress并组装成transition字典
                current_group = []
                total_steps = len(observations_list)
                
                for i in range(total_steps):
                    progress = i / max(1, total_steps - 1) if total_steps > 1 else 0.0
                    
                    transition = {
                        "progress": progress,
                        "observations": observations_list[i],
                        "actions": actions_list[i],
                        "next_observations": next_observations_list[i],
                        "masks": masks_list[i],
                        "dones": dones_list[i],
                    }
                    current_group.append(transition)
                
                groups.append(current_group)
                pbar.update(1)
                print(f"保存了包含 {len(current_group)} 步的组")
            else:
                print(f"丢弃了包含 {len(observations_list)} 步的组")
            
            # 重置环境和数据列表
            obs, _ = env.reset()
            observations_list = []
            actions_list = []
            next_observations_list = []
            masks_list = []
            dones_list = []
            
            group_idx += 1
            reset_key = False
            save_and_reset_key = False
            
            if done or truncated:
                time.sleep(1.5)

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_grouped_data_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(groups, f)
        print(f"saved {len(groups)} groups to {file_name}")
        
if __name__ == "__main__":
    app.run(main)
