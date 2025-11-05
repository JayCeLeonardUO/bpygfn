from gfn_environments.single_color_ramp import *

import torch


# forward policy

EPOCHS = 100

REPLAY_BUFFER_SIZE = 1000
TRAJECTORY_LEN = 10

from gfn_environments.single_color_ramp import v2StepWEnv

blender_api = BlenderTerrainAPI()
# this is what we use to get "tensorfied" env actions
env_serializer = v2StepWEnv()

import random

def random_polocy(env_s:v2StepWEnv, num_samples):
    """
    used to prefill the buffer.
    will sample random action
    and terminate them with stop
    """
    num_actions = len(env_s.tensor_to_action)

    traj_list= []

    action_tensors = list(env_s.tensor_to_action.keys())
    heightmaps = []
    for i in range(num_samples):
        blender_api.reset_env()
        _trajectory= []
        for step in range(TRAJECTORY_LEN-1):
            _random_action = random.sample(action_tensors,1)[0]

            _one_hot  = torch.zeros(num_actions)
            _one_hot[_random_action] = 1.0

            env_s.execute_one_hot_action(blender_api,_one_hot)

            _trajectory.append(_one_hot)

        # I just know that the stop action is the last one

        _trajectory.append(action_tensors[-1])
        _heightmap = blender_api.get_heightmap()
        heightmaps.append(_heightmap)
        traj_list.append(_trajectory)

    return traj_list, heightmaps


replay_buffer_cap = 5
# prefill the replay buffer
import pandas as pd

prefill_trajectories, prefill_heightmaps = random_polocy(env_serializer,replay_buffer_cap)

replay_buffer = pd.DataFrame()

replay_buffer['trajectorys'] = prefill_trajectories
replay_buffer['heightmaps'] = prefill_heightmaps

#%ln[0]:


