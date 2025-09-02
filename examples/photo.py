import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
from absl import app, flags
from pynput import keyboard
from experiments.move_bread.config import TrainConfig as OpenDrawerTrainConfig

FLAGS = flags.FLAGS
flags.DEFINE_integer("imgs_needed", 1, "Number of successful images to collect.")

save_key = False

def on_press(key):
    global save_key
    try:
        if key == keyboard.Key.space:
            save_key = True
            print("Space Key Pressed! (保存图像)")
    except AttributeError:
        pass
    
def main(_):
    global save_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    config = OpenDrawerTrainConfig()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    
    obs, _ = env.reset()
    
    cnt = 0
    while cnt < 20:
        actions = np.zeros(env.action_space.sample().shape)
        actions[6] = 1
        obs, rew, done, truncated, info = env.step(actions)
        if save_key:
            img_path = f"/home/kkk/workspace/hil-serl/photos/photo{cnt}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            for key in obs:
                if key in ["wrist"]:
                    img_data = obs[key][0]
                    img = Image.fromarray(img_data.astype(np.uint8))
                    img.save(img_path)    
                    cnt += 1
                    save_key = False
                    print(f"已保存图像: {img_path}，key: {key}")
                else:
                    print("警告: 未找到可保存的图像")
                    
if __name__ == "__main__":
    app.run(main)