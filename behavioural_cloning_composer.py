# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.
# NOTE: This is _not_ the original code used for VPT!
#       This is merely to illustrate how to fine-tune the models and includes
#       the processing steps used.

# This will likely be much worse than what original VPT did:
# we are not training on full sequences, but only one step at a time to save VRAM.

from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch as th
import numpy as np

from composer.core import DataSpec
from composer import Trainer
from agent_composer import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from lib.tree_util import tree_map

EPOCHS = 2
# Needs to be <= number of videos
BATCH_SIZE = 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 12
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs
        
def num_samples_fn(batch):
    return BATCH_SIZE
  
def behavioural_cloning_train(data_dir, in_model, in_weights, out_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    policy = agent.policy
    trainable_parameters = policy.parameters()

    
    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS
    )
    
    data_spec = DataSpec(data_loader, get_num_samples_in_batch=num_samples_fn,)
    
    trainer = Trainer(
        model=agent,
        optimizers=optimizer,
        train_dataloader=data_spec,
        max_duration='10ep'
    )
    
    
    trainer.fit()

    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)
