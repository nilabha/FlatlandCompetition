import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
import sys
import time

# from evostra import EvolutionStrategy
#from pytorch_es import EvolutionModule
from strategies.evolution import EvolutionModule
from pytorch_es.utils.helpers import weights_init
import gym
from gym import logger as gym_logger
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn

import getopt
import random
import sys
from collections import deque

# make sure the root path is in system path
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_training.dueling_double_dqn import Agent

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus


gym_logger.setLevel(logging.CRITICAL)

training=True
cuda = "cpu" #torch.cuda.is_available()

random.seed(1)
np.random.seed(1)

# Parameters for the Environment
x_dim = 35
y_dim = 35
n_agents = 2

weightsFile = f'weightsESAgent{n_agents}.p'

# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.05,  # Percentage of defective agents
                   'malfunction_rate': 100,  # Rate of malfunction occurence
                   'min_duration': 20,  # Minimal duration of malfunction
                   'max_duration': 50  # Max duration of malfunction
                   }

# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We set the number of episodes we would like to train on
if 'n_trials' not in locals():
    n_trials = 15000

# env_Orig = RailEnv(width=x_dim,
#               height=y_dim,
#               rail_generator=sparse_rail_generator(max_num_cities=3,
#                                                    # Number of cities in map (where train stations are)
#                                                    seed=1,  # Random seed
#                                                    grid_mode=False,
#                                                    max_rails_between_cities=2,
#                                                    max_rails_in_city=3),
#               schedule_generator=sparse_schedule_generator(speed_ration_map),
#               number_of_agents=n_agents,
#               stochastic_data=stochastic_data,  # Malfunction data generator
#               obs_builder_object=TreeObservation)

# After training we want to render the results so we also load a renderer
# env_renderer = RenderTool(env, gl="PILSVG", )
# Given the depth of the tree observation and the number of features per node we get the following state_size
num_features_per_node = 11 #env_Orig.obs_builder.observation_dim
tree_depth = 2
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes

#print("State Size:",state_size)
#print("Feature Size:", num_features_per_node)
# The action space of flatland is 5 discrete actions
action_size = 5
hidden_size = 100
# add the model on top of the convolutional base
model = nn.Sequential(
    nn.Linear(state_size, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, action_size),
    nn.Softmax()
)

# model.apply(weights_init)

if cuda:
    model = model.cuda()

columns = ['Agents', 'X_DIM', 'Y_DIM', 'TRIAL_NO', 'SCORE',
           'DONE_RATIO', 'STEPS', 'ACTION_PROB']
dfAllResults = pd.DataFrame(columns=columns)

# And some variables to keep track of the progress
action_dict = dict()
final_action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
scores = []
dones_list = []
action_prob = [0] * action_size
agent_obs = [None] * n_agents
agent_next_obs = [None] * n_agents
agent_obs_buffer = [None] * n_agents
agent_action_buffer = [2] * n_agents
cummulated_reward = np.zeros(n_agents)
update_values = [False] * n_agents


def get_reward(weights, model, render=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)


    env_Orig = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=3,
                                                       # Number of cities in map (where train stations are)
                                                       seed=1,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=TreeObservation)

    env = copy.deepcopy(env_Orig)

    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG", )

    # And the max number of steps we want to take per episode
    max_steps = int(4 * 2 * (20 + env.height + env.width))

    n_episodes = 1
    for trials in range(1, n_episodes + 1):
        # Reset environment
        obs, info = env.reset(True, True)
        env_renderer.reset()
        # Build agent specific observations
        for a in range(env.get_num_agents()):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
                agent_obs_buffer[a] = agent_obs[a].copy()

        # Reset score and done
        score = 0
        env_done = 0
        step = 0

        # Run episode
        while True:
            # Action
            for a in range(env.get_num_agents()):
                if info['action_required'][a]:
                    # If an action is require, we want to store the obs a that step as well as the action
                    update_values[a] = True

                    batch = torch.from_numpy(agent_obs[a][np.newaxis, ...]).float()
                    if cuda:
                        batch = batch.cuda()
                    prediction = cloned_model(Variable(batch))
                    action = prediction.data.cpu().numpy().argmax()

                    # action = agent.act(agent_obs[a], eps=eps)
                    action_prob[action] += 1
                else:
                    update_values[a] = False
                    action = 0
                action_dict.update({a: action})

            # Environment step
            # print("Action Values:", action_dict)
            next_obs, all_rewards, done, info = env.step(action_dict)
            step+=1
            if (render):
                env_renderer.render_env(show=True, show_predictions=True, show_observations=False)

            for a in range(env.get_num_agents()):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values[a] or done[a]:
                    # agent.step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a],
                    #           agent_obs[a], done[a])
                    cummulated_reward[a] = 0.

                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]
                if next_obs[a]:
                    agent_obs[a] = normalize_observation(next_obs[a], tree_depth, observation_radius=10)

                score += all_rewards[a] / env.get_num_agents()
            # print(all_rewards)
            # Copy observation
            if done['__all__'] or step >= max_steps:
                env_done = 1
                break

        # Collection information about training
        tasks_finished = 0
        for current_agent in env.agents:
            if current_agent.status == RailAgentStatus.DONE_REMOVED:
                tasks_finished += 1
        done_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                action_prob / np.sum(action_prob)), end=" ")

    # env.close()
    data = [[n_agents, x_dim, y_dim,
             trials,
             np.mean(scores_window),
             100 * np.mean(done_window),
             step, action_prob / np.sum(action_prob)]]

    dfCur = pd.DataFrame(data)

    with open(f'ES_TrainingResults_{n_agents}_{x_dim}_{y_dim}.csv', 'a') as f:
        dfCur.to_csv(f, index=False,header=False)

    return np.mean(scores)

partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=5, sigma=0.1,
    learning_rate=0.001, threadcount=1, cuda=cuda, reward_goal=1,
    consecutive_goal_stopping=10,save_path=weightsFile
)
start = time.time()

if training:
    final_weights = es.run(500,print_step=1)
    pickle.dump(final_weights, open(os.path.abspath(weightsFile), 'wb'))

end = time.time() - start
final_weights = pickle.load(open(os.path.abspath(weightsFile), 'rb'))

reward = partial_func(final_weights, render=True)
print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")

