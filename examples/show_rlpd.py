#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

def init(configname, start_step, checkpoint_path, seed=42):

    config = configname()
    rng = jax.random.PRNGKey(seed)
    rng, sampling_rng = jax.random.split(rng)

    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
        seed=seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )

    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    ckpt = checkpoints.restore_checkpoint(
        checkpoint_path,
        agent.state,
        step=start_step,
        )
    agent = agent.replace(state=ckpt)
    
    ckpt = checkpoints.restore_checkpoint(
        checkpoint_path,
        agent.state,
        step=start_step,
    )
    agent = agent.replace(state=ckpt)
    
    return sampling_rng, agent, env
    
def task(sampling_rng, agent, env):
    success = False
    while not success:
        obs, _ = env.reset()
        done = False
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=False,
                seed=key
            )
            actions = np.asarray(jax.device_get(actions))

            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs

            if done and (reward >= 3.5):
                success = True
    print("\033[93m {}\033[0m".format("Task SUCCESS"))
    