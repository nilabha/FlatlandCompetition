from flatland.evaluators.client import FlatlandRemoteClient
from flatland.core.env_observation_builder import DummyObservationBuilder
from my_observation_builder import CustomObservationBuilder
import numpy as np
import time

import getopt
import random
import sys
from typing import List

import os
from itertools import combinations

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params

from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file

from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.envs.agent_utils import RailAgentStatus, EnvAgent

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from flatland.utils.ordered_set import OrderedSet

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet


os.environ["AICROWD_TESTS_FOLDER"] ="./test-envs/"

class MultipleAgentNavigationObs(TreeObsForRailEnv):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self,max_depth: int, predictor: PredictionBuilder = None):
        super().__init__(max_depth,predictor)

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        distance_map = self.env.distance_map.get()

        visited = set()
        for _idx in range(30):
            # Check if any of the other prediction overlap with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))

        # This variable will be access by the renderer to visualize the observation
        self.env.dev_obs_dict[handle] = visited


        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = get_new_position(agent_virtual_position, direction)
                min_distances.append(
                    distance_map[handle, new_position[0], new_position[1], direction])
            else:
                min_distances.append(np.inf)

        if num_transitions == 1:
            observation = [0, 1, 0]
            observation = np.tile(observation, 3)

        elif num_transitions == 0:
            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1
            observation = np.tile(observation, 3)

        elif num_transitions == 2:
            idx = np.argpartition(np.array(min_distances), 2)
            observation1 = [0, 0, 0]
            observation1[idx[0]] = 1

            observation2 = [0, 0, 0]
            observation2[idx[1]] = 1

            observation = np.hstack([observation1, observation2, observation1])

        elif num_transitions == 3:
            idx = np.argpartition(np.array(min_distances), 3)
            observation1 = [0, 0, 0]
            observation1[idx[0]] = 1

            observation2 = [0, 0, 0]
            observation2[idx[1]] = 1

            observation3 = [0, 0, 0]
            observation3[idx[2]] = 1

            observation = np.hstack([observation1, observation2, observation3])

        min_distances = np.sort(min_distances)
        incremental_distances = np.diff(np.sort(min_distances))
        incremental_distances[incremental_distances == np.inf] = -1
        incremental_distances[np.isnan(incremental_distances)] = -1
        min_distances[min_distances == np.inf] = -1
        observation = np.hstack([observation, incremental_distances])

        observation = np.hstack([distance_map[(handle, *agent_virtual_position,
                                                            agent.direction)], observation,
                                                            agent.malfunction_data['malfunction'],
                                                            agent.speed_data['speed'],agent.speed_data['position_fraction']])

        return observation,self.predictions[handle], self.predicted_pos

#####################################################################
# Instantiate a Remote Client
#####################################################################
remote_client = FlatlandRemoteClient()

#####################################################################
# Define your custom controller
#
# which can take an observation, and the number of agents and 
# compute the necessary action for this step for all (or even some)
# of the agents
#####################################################################
def my_controller(obs, number_of_agents):
    _action = {}
    for _idx in range(number_of_agents):
        _action[_idx] = np.random.randint(0, 5)
    return _action

#####################################################################
# Instantiate your custom Observation Builder
# 
# You can build your own Observation Builder by following 
# the example here : 
# https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/observations.py#L14
#####################################################################
my_observation_builder = MultipleAgentNavigationObs(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

# Or if you want to use your own approach to build the observation from the env_step, 
# please feel free to pass a DummyObservationBuilder() object as mentioned below,
# and that will just return a placeholder True for all observation, and you 
# can build your own Observation for all the agents as your please.
# my_observation_builder = DummyObservationBuilder()


#####################################################################
# Main evaluation loop
#
# This iterates over an arbitrary number of env evaluations
#####################################################################
evaluation_number = 0
while True:

    evaluation_number += 1
    # Switch to a new evaluation environemnt
    # 
    # a remote_client.env_create is similar to instantiating a 
    # RailEnv and then doing a env.reset()
    # hence it returns the first observation from the 
    # env.reset()
    # 
    # You can also pass your custom observation_builder object
    # to allow you to have as much control as you wish 
    # over the observation of your choice.
    time_start = time.time()
    observation, info = remote_client.env_create(
                    obs_builder_object=my_observation_builder
                )
    env_creation_time = time.time() - time_start
    if not observation:
        #
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been 
        # evaluated on all the required evaluation environments,
        # and hence its safe to break out of the main evaluation loop
        break
    
    print("Evaluation Number : {}".format(evaluation_number))

    #####################################################################
    # Access to a local copy of the environment
    # 
    #####################################################################
    # Note: You can access a local copy of the environment 
    # by using : 
    #       remote_client.env 
    # 
    # But please ensure to not make any changes (or perform any action) on 
    # the local copy of the env, as then it will diverge from 
    # the state of the remote copy of the env, and the observations and 
    # rewards, etc will behave unexpectedly
    # 
    # You can however probe the local_env instance to get any information
    # you need from the environment. It is a valid RailEnv instance.
    local_env = remote_client.env
    number_of_agents = len(local_env.agents)
    n_agents = number_of_agents
    env = local_env
    obs = observation
    # Now we enter into another infinite loop where we 
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    # 
    # An episode is considered done when either all the agents have 
    # reached their target destination
    # or when the number of time steps has exceed max_time_steps, which 
    # is defined by : 
    #
    # max_time_steps = int(4 * 2 * (env.width + env.height + 20))
    #
    time_taken_by_controller = []
    time_taken_per_step = []
    steps = 0

    # Reset score and done
    score = 0
    env_done = 0
    step = 0
    max_steps = int(4 * 2 * (20 + env.height + env.width))

    while True:
        #####################################################################
        # Evaluation of a single episode
        #
        #####################################################################
        # Compute the action for this step by using the previously 
        # defined controller
        time_start = time.time()
        #action = my_controller(observation, number_of_agents)

        for i in range(n_agents):
            if obs[i] is not None:
                observations, prediction_data, prediction_pos = obs[i]
                break

        action_dict = {}
        next_shortest_actions = 2 * np.ones(n_agents)
        next_next_shortest_actions = 2 * np.ones(n_agents)
        agent_conflicts = np.zeros((n_agents, n_agents))
        agent_conflicts_count = np.zeros((n_agents, n_agents))
        minDist = -1 * np.ones(n_agents)
        incDiff1 = -1 * np.ones(n_agents)
        incDiff2 = -1 * np.ones(n_agents)
        malfunc = np.zeros(n_agents)
        speed = np.ones(n_agents)
        pos_frac = np.ones(n_agents)
        agent_num_conflicts = []

        vals = []
        counts = []
        counter = np.zeros(n_agents)
        for i in range(30):
            pos = prediction_pos[i]
            val, count = np.unique(pos, return_counts=True)
            if (val[0] == -1):
                val = val[1:]
                count = count[1:]
            vals.append(val)
            counts.append(count)

            for j, curVal in enumerate(val):
                # curVal = vals[i]
                curCount = count[j]
                if curCount > 1:
                    idxs = np.argwhere(pos == curVal)
                    lsIdx = [int(x) for x in idxs]
                    combs = list(combinations(lsIdx, 2))
                    for k, comb in enumerate(combs):
                        counter[comb[0]] += 1
                        counter[comb[1]] += 1
                        agent_conflicts_count[comb[0], comb[1]] = (counter[comb[0]] + counter[comb[1]]) / 2
                        if agent_conflicts[comb[0], comb[1]] == 0:
                            agent_conflicts[comb[0], comb[1]] = i
                        else:
                            agent_conflicts[comb[0], comb[1]] = min(i, agent_conflicts[comb[0], comb[1]])

        for i in range(n_agents):
            agent_num_conflicts.append(sum(agent_conflicts[i, :]))
            if not obs or obs is None or obs[i] is None:
                action_dict.update({i: 2})
            elif obs[i][0] is not None:
                shortest_action = np.argmax(obs[i][0][1:4]) + 1
                next_shortest_action = np.argmax(obs[i][0][5:7]) + 1
                next_next_shortest_action = np.argmax(obs[i][0][8:10]) + 1
                next_shortest_actions[i] = next_shortest_action
                next_next_shortest_actions[i] = next_next_shortest_action
                malfunc[i] = obs[i][0][-3]
                speed[i] = obs[i][0][-2]
                pos_frac[i] = obs[i][0][-1]
                minDist[i] = obs[i][0][0]
                incDiff1[i] = obs[i][0][-5]
                incDiff2[i] = obs[i][0][-4]
                action_dict.update({i: shortest_action})
            else:
                action_dict.update({i: 2})
        mal_agents = (np.array(-1))
        for i in range(n_agents):
            if agent_num_conflicts[i] > 0:
                mal_agents = np.where(malfunc > 0)
                for i, mal_agent in enumerate(mal_agents[0]):
                    if mal_agent is None:
                        break
                    conflict_agents = np.where(agent_conflicts[:, int(mal_agent)] > 0)

                    for j, cur_conflict_agent in enumerate(conflict_agents[0]):
                        cur_conflict_agent = int(cur_conflict_agent)
                        steps_conflict = agent_conflicts[cur_conflict_agent, mal_agent]
                        if steps_conflict <= 3:
                            if incDiff1[cur_conflict_agent] == -1:
                                if int(minDist[cur_conflict_agent]) >= 5:
                                    action_dict.update({cur_conflict_agent: 4})
                                elif agent_conflicts_count[cur_conflict_agent, mal_agent] > 1:
                                    action_dict.update({cur_conflict_agent: 4})
                            elif minDist[cur_conflict_agent] > incDiff1[cur_conflict_agent]:
                                action_dict.update({cur_conflict_agent: 4})
                            else:
                                action_dict.update({cur_conflict_agent: next_shortest_actions[cur_conflict_agent]})

        time_taken = time.time() - time_start
        time_taken_by_controller.append(time_taken)

        # Perform the chosen action on the environment.
        # The action gets applied to both the local and the remote copy
        # of the environment instance, and the observation is what is
        # returned by the local copy of the env, and the rewards, and done and info
        # are returned by the remote copy of the env
        time_start = time.time()
        observation, all_rewards, done, info = remote_client.env_step(action_dict)
        
        #print("Rewards: ", all_rewards, "  [done=", done, "]")

        for a in range(env.get_num_agents()):
            score += all_rewards[a] / env.get_num_agents()
        
        steps += 1
        time_taken = time.time() - time_start
        time_taken_per_step.append(time_taken)

        if done['__all__'] or steps >= max_steps:
            print("Reward : ", sum(list(all_rewards.values())))
            #
            # When done['__all__'] == True, then the evaluation of this
            # particular Env instantiation is complete, and we can break out
            # of this loop, and move onto the next Env evaluation
            break

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("="*100)
    print("="*100)
    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)
    print("Number of Steps : ", steps)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("="*100)

print("Evaluation of all environments complete...")
########################################################################
# Submit your Results
# 
# Please do not forget to include this call, as this triggers the 
# final computation of the score statistics, video generation, etc
# and is necesaary to have your submission marked as successfully evaluated
########################################################################
print(remote_client.submit())
