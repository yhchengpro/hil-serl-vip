import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("imgs_needed", 20, "Number of successful images to collect.")


import faulthandler
faulthandler.enable()

reset_key = False
save_key = False
break_key = False

def on_press(key):
    global reset_key, break_key, save_key
    try:
        if hasattr(key, 'char') and key.char in ['r', 'R']:
            reset_key = True
            print("Reset Key Pressed! (复位)")
        elif key == keyboard.Key.esc:
            break_key = True
            print("Break Key Pressed! (退出)")
        elif key == keyboard.Key.space:
            save_key = True
            print("Space Key Pressed! (保存图像)")
    except AttributeError:
        pass

def main(_):
    global reset_key, break_key, save_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()

    cnt = 0

    while cnt < FLAGS.imgs_needed:
        actions = np.zeros(env.action_space.sample().shape)
        obs, rew, done, truncated, info = env.step(actions)
        #print(f"obs{obs}")
        
        if save_key:
            # 保存当前观察到的图像
            img_path = f"./experiments/{FLAGS.exp_name}/goal_images/goal_image_{cnt}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            for key in obs:
                if key in config.reward_keys:
                    img_data = obs[key][0]
                    img = Image.fromarray(img_data.astype(np.uint8))
                    img.save(img_path)            
                    cnt += 1
                    save_key = False
                    print(f"已保存图像: {img_path}，key: {key}")
                else:
                    print("警告: 未找到可保存的图像")

        if reset_key:
            env.reset()
            print("环境已重置")
            time.sleep(1)
            reset_key = False

        if break_key:
            env.close()
            print("退出程序")
            break
        
if __name__ == "__main__":
    app.run(main)
