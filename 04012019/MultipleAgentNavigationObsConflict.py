import getopt
import random
import sys
import time
from typing import List

import numpy as np
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
random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)


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


def main(args):
    try:
        opts, args = getopt.getopt(args, "", ["sleep-for-animation=", ""])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)
    sleep_for_animation = True
    for o, a in opts:
        if o in ("--sleep-for-animation"):
            sleep_for_animation = str2bool(a)
        else:
            assert False, "unhandled option"

    test_envs_root = "./railway"
    test_env_file_path = "testing_stuff.pkl"

    test_env_file_path = os.path.join(
        test_envs_root,
        test_env_file_path
    )

    x_dim = 7
    y_dim = 7
    n_agents = 3

    stochastic_data = {'prop_malfunction': 0.05,  # Percentage of defective agents
                       'malfunction_rate': 100,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # env = RailEnv(width=1, height=1, rail_generator=rail_from_file(test_env_file_path),
    #                    schedule_generator=schedule_from_file(test_env_file_path),
    #                    #malfunction_generator_and_process_data=malfunction_from_file(test_env_file_path),
    #                    obs_builder_object=MultipleAgentNavigationObs(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30)))
    #
    # #env.number_of_agents = n_agents
    # n_agents = env.number_of_agents
    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=6, max_dist=99999,seed=1),
                  # sparse_rail_generator(max_num_cities=3,
                  #                                      # Number of cities in map (where train stations are)
                  #                                      seed=1,  # Random seed
                  #                                      grid_mode=False,
                  #                                      max_rails_between_cities=2,
                  #                                      max_rails_in_city=3),
                  schedule_generator=complex_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
    #
    # env = RailEnv(width=7, height=7,
    #               rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999,
    #                                                     seed=1), schedule_generator=complex_schedule_generator(),
    #               number_of_agents=n_agents,
                  obs_builder_object=MultipleAgentNavigationObs(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30)))

    max_steps = int(4 * 2 * (20 + env.height + env.width))
    obs, info = env.reset(regenerate_rail=True,
            regenerate_schedule=True,
            random_seed=random_seed)

    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env(show=True, frames=True, show_observations=True)

    # Reset score and done
    score = 0
    env_done = 0
    step = 0
    for step in range(max_steps):

        for i in range(n_agents):
            if obs[i] is not None:
                observations, prediction_data, prediction_pos = obs[i]
                break

        action_dict = {}
        next_shortest_actions = 2*np.ones(n_agents)
        next_next_shortest_actions = 2*np.ones(n_agents)
        agent_conflicts = np.zeros((n_agents,n_agents))
        agent_conflicts_count = np.zeros((n_agents, n_agents))
        minDist = -1 *np.ones(n_agents)
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
            if(val[0] == -1):
                val = val[1:]
                count = count[1:]
            vals.append(val)
            counts.append(count)

            for j,curVal in enumerate(val):
                #curVal = vals[i]
                curCount = count[j]
                if curCount > 1:
                    idxs = np.argwhere(pos == curVal)
                    lsIdx = [int(x) for x in idxs]
                    combs = list(combinations(lsIdx,2))
                    for k,comb in enumerate(combs):
                        counter[comb[0]] += 1
                        counter[comb[1]] += 1
                        agent_conflicts_count[comb[0], comb[1]] = (counter[comb[0]] + counter[comb[1]])/2
                        if agent_conflicts[comb[0], comb[1]] == 0:
                            agent_conflicts[comb[0], comb[1]] = i
                        else:
                            agent_conflicts[comb[0], comb[1]] = min(i, agent_conflicts[comb[0], comb[1]])

        for i in range(n_agents):
            agent_num_conflicts.append(sum(agent_conflicts[i,:]))
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
                for i,mal_agent in enumerate(mal_agents[0]):
                    if mal_agent is None:
                        break
                    conflict_agents = np.where(agent_conflicts[:,int(mal_agent)]>0)

                    for j,cur_conflict_agent in enumerate(conflict_agents[0]):
                        cur_conflict_agent = int(cur_conflict_agent)
                        steps_conflict = agent_conflicts[cur_conflict_agent, mal_agent]
                        if steps_conflict <= 3:
                            if incDiff1[cur_conflict_agent] == -1:
                                if int(minDist[cur_conflict_agent]) >= 5:
                                    action_dict.update({cur_conflict_agent: 4})
                                elif agent_conflicts_count[cur_conflict_agent,mal_agent] > 1:
                                    action_dict.update({cur_conflict_agent: 4})
                            elif minDist[cur_conflict_agent] > incDiff1[cur_conflict_agent]:
                                action_dict.update({cur_conflict_agent: 4})
                            else:
                                action_dict.update({cur_conflict_agent: next_shortest_actions[cur_conflict_agent]})

        obs, all_rewards, done, _ = env.step(action_dict)

        print("Rewards: ", all_rewards, "  [done=", done, "]")

        for a in range(env.get_num_agents()):
            score += all_rewards[a] / env.get_num_agents()

        env_renderer.render_env(show=True, frames=True, show_observations=True)
        if sleep_for_animation:
            time.sleep(0.5)
        if done["__all__"]:
            break

        # Collection information about training
        tasks_finished = 0
        for current_agent in env.agents:
            if current_agent.status == RailAgentStatus.DONE_REMOVED:
                tasks_finished += 1
        done_window = tasks_finished / max(1, env.get_num_agents())
        scores_window = score / max_steps
        print(
            '\rTraining {} Agents on ({},{}).\t Steps {}\t Average Score: {:.3f}\tDones: {:.2f}%\t'.format(
                env.get_num_agents(), x_dim, y_dim,
                step,
                np.mean(scores_window),
                100 * np.mean(done_window)), end=" ")

    tasks_finished = 0
    for current_agent in env.agents:
        if current_agent.status == RailAgentStatus.DONE_REMOVED:
            tasks_finished += 1
    done_window = tasks_finished / max(1, env.get_num_agents())
    scores_window = score / max_steps
    print(
        '\rTraining {} Agents on ({},{}).\t Total Steps {}\t Average Score: {:.3f}\tDones: {:.2f}%\t'.format(
            env.get_num_agents(), x_dim, y_dim,
            step,
            np.mean(scores_window),
            100 * np.mean(done_window)), end=" ")

    env_renderer.close_window()


if __name__ == '__main__':
    if 'argv' in globals():
        main(sys.argv)
    else:
        main(sys.argv[1:])
