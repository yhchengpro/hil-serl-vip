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
flags.DEFINE_integer("successes_needed", 100, "Number of successful transistions to collect.")


success_key = False
reset_key = False
def on_press(key):
    global success_key, reset_key
    try:
        if key == keyboard.Key.space:
            success_key = not success_key
            print("Success Key Pressed!")
        elif hasattr(key, 'char') and key.char in ['r', 'R']:
            reset_key = True
            print("Reset Key Pressed!")
    except AttributeError:
        pass

def main(_):
    global success_key, reset_key
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape)
        # actions[6] = 1
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        print(f"status: {success_key}, {np.round(obs['state'][0],1)}")
        if success_key:# and obs['state'][0][0] == -1:
            successes.append(transition)
            pbar.update(1)
        else:
            failures.append(transition)
        if reset_key:
            success_key = False
            reset_key = False
            obs, _ = env.reset()
            reset_key = False

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./experiments/{FLAGS.exp_name}/classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./experiments/{FLAGS.exp_name}/classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
    obs, _ = env.reset()
    env.close()
        
if __name__ == "__main__":
    app.run(main)
