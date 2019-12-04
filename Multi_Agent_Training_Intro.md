# How to train multiple Agents on Flatland
Quick introduction on how to train a simple DQN agent using Flatland and Pytorch. At the end of this Tutorial you should be able to train a single agent to navigate in Flatland.
We use the `multi_agent_training.py` ([here](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/multi_agent_training.py)) file to train multiple agents on the avoid conflicts task.

## Actions in Flatland
Flatland is a railway simulation. Thus the actions of an agent are strongly limited to the railway network. This means that in many cases not all actions are valid.
The possible actions of an agent are

- 0 *Do Nothing*:  If the agent is moving it continues moving, if it is stopped it stays stopped
- 1 *Deviate Left*: This action is only valid at cells where the agent can change direction towards left. If action is chosen, the left transition and a rotation of the agent orientation to the left is executed. If the agent is stopped at any position, this action will cause it to start moving in any cell where forward or left is allowed!
- 2 *Go Forward*: This action will start the agent when stopped. At switches this will chose the forward direction.
- 3 *Deviate Right*: Exactly the same as deviate left but for right turns.
- 4 *Stop*: This action causes the agent to stop, this is necessary to avoid conflicts in multi agent setups (Not needed for navigation).

## Shortest path predictor
With multiple agents alot of conlflicts will arise on the railway network. These conflicts arise because different agents want to occupie the same cells at the same time. Due to the nature of the railway network and the dynamic of the railway agents (can't turn around), the conflicts have to be detected in advance in order to avoid them. If agents are facing each other and don't have any options to deviate from their path it is called a *deadlock*.
Therefore we introduce a simple prediction function that predicts the most likely (here shortest) path of all the agents. Furthermore, the prediction is withdrawn if an agent stopps and replaced by a prediction that the agent will stay put. The predictions allow the agents to detect possible conflicts before they happen and thus performe counter measures.
*ATTENTION*: This is a very basic implementation of a predictor. It will not solve all the problems because it always predicts shortest paths and not alternative routes. It is up to you to come up with much more clever predictors to avod conflicts!

## Tree Observation
Flatland offers three basic observations from the beginning. We encourage you to develop your own observations that are better suited for this specific task.

For the navigation training we start with the Tree Observation as agents will learn the task very quickly using this observation.
The tree observation exploits the fact that a railway network is a graph and thus the observation is only built along allowed transitions in the graph.

Here is a small example of a railway network with an agent in the top left corner. The tree observation is build by following the allowed transitions for that agent.

![Small_Network](https://i.imgur.com/utqMx08.png)

As we move along the allowed transitions we build up a tree where a new node is created at every cell where the agent has different possibilities (Switch), dead-end or the target is reached.
It is important to note that the tree observation is always build according to the orientation of the agent at a given node. This means that each node always has 4 branches coming from it in the directions *Left, Forward, Right and Backward*. These are illustrated with different colors in the figure below. The tree is build form the example rail above. Nodes where there are no possibilities are filled with `-inf` and are not all shown here for simplicity. The tree however, always has the same number of nodes for a given tree depth.

![Tree_Observation](https://i.imgur.com/VsUQOQz.png)

### Node Information
Each node is filled with information gathered along the path to the node. Currently each node contains 9 features:

- 1: if own target lies on the explored branch the current distance from the agent in number of cells is stored.

- 2: if another agents target is detected the distance in number of cells from current agent position is stored.

- 3: if another agent is detected the distance in number of cells from current agent position is stored.

- 4: possible conflict detected (This only works when we use a predictor and will not be important in this tutorial)


- 5: if an not usable switch (for agent) is detected we store the distance. An unusable switch is a switch where the agent does not have any choice of path, but other agents coming from different directions might. 


- 6: This feature stores the distance (in number of cells) to the next node (e.g. switch or target or dead-end)

- 7: minimum remaining travel distance from node to the agent's target given the direction of the agent if this path is chosen


- 8: agent in the same direction found on path to node
    - n = number of agents present same direction (possible future use: number of other agents in the same direction in this branch)
    - 0 = no agent present same direction

- 9: agent in the opposite direction on path to node
    - n = number of agents present other direction than myself
    - 0 = no agent present other direction than myself

For training purposes the tree is flattend into a single array.


## Training
### Setting up the environment
Let us now train a simle double dueling DQN agent to detect to find its target and try to avoid conflicts on flatland. We start by importing the necessary packages from Flatland. Note that we now also import a predictor from `flatland.envs.predictions`

```
from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from utils.observation_utils import norm_obs_clip, split_tree
```

For this simple example we want to train on randomly generated levels using the `complex_rail_generator`. The training curriculum will use different sets of parameters throughout training to enhance generalizability of the solution.

```
# Initialize a random map with a random number of agents
x_dim = np.random.randint(8, 20)
y_dim = np.random.randint(8, 20)
n_agents = np.random.randint(3, 8)
n_goals = n_agents + np.random.randint(0, 3)
min_dist = int(0.75 * min(x_dim, y_dim))
tree_depth = 3
```

As mentioned above, for this experiment we are going to use the tree observation and thus we load the observation builder. Also we are now using the predictor as well which is passed to the observation builder.

```
"""
 Get an observation builder and predictor:
 The predictor will always predict the shortest path from the current location of the agent.
 This is used to warn for potential conflicts --> Should be enhanced to get better performance!
"""
predictor = ShortestPathPredictorForRailEnv()
observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=predictor)
```

And pass it as an argument to the environment setup

```
env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                    max_dist=99999,
                                                    seed=0),
              obs_builder_object=observation_builder,
              number_of_agents=n_agents)
```

We have no successfully set up the environment for training. To visualize it in the renderer we also initiate the renderer with.

###Setting up the agent

To set up a appropriate agent we need the state and action space sizes. From the discussion above about the tree observation we end up with:


```
num_features_per_node = env.obs_builder.observation_dim
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes
action_size = 5
```

In the `multi_agent_training.py` file you will find further variable that we initiate in order to keep track of the training progress.
Below you see an example code to train an agent. It is important to note that we reshape and normalize the tree observation provided by the environment to facilitate training.
To do so, we use the utility functions `split_tree(tree=np.array(obs[a]), num_features_per_node=features_per_node, current_depth=0)` and `norm_obs_clip()`. Feel free to modify the normalization as you see fit.

```
# Split the observation tree into its parts and normalize the observation using the utility functions.
    # Build agent specific local observation
    for a in range(env.get_num_agents()):
        rail_data, distance_data, agent_data = split_tree(tree=np.array(obs[a]),
                                                          num_features_per_node=features_per_node,
                                                          current_depth=0)
        rail_data = norm_obs_clip(rail_data)
        distance_data = norm_obs_clip(distance_data)
        agent_data = np.clip(agent_data, -1, 1)
        agent_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))
```

We now use the normalized `agent_obs` for our training loop:


```
# Do training over n_episodes
    for episodes in range(1, n_episodes + 1):
        """
        Training Curriculum: In order to get good generalization we change the number of agents
        and the size of the levels every 50 episodes.
        """
        if episodes % 50 == 0:
            x_dim = np.random.randint(8, 20)
            y_dim = np.random.randint(8, 20)
            n_agents = np.random.randint(3, 8)
            n_goals = n_agents + np.random.randint(0, 3)
            min_dist = int(0.75 * min(x_dim, y_dim))
            env = RailEnv(width=x_dim,
                          height=y_dim,
                          rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                                max_dist=99999,
                                                                seed=0),
                          obs_builder_object=observation_helper,
                          number_of_agents=n_agents)

            # Adjust the parameters according to the new env.
            max_steps = int(3 * (env.height + env.width))
            agent_obs = [None] * env.get_num_agents()
            agent_next_obs = [None] * env.get_num_agents()

        # Reset environment
        obs, info = env.reset(True, True)

        # Setup placeholder for finals observation of a single agent. This is necessary because agents terminate at
        # different times during an episode
        final_obs = agent_obs.copy()
        final_obs_next = agent_next_obs.copy()

        # Build agent specific observations
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree(tree=np.array(obs[a]), num_features_per_node=num_features_per_node,
                                                    current_depth=0)
            data = norm_obs_clip(data, fixed_radius=observation_radius)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            agent_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        score = 0
        env_done = 0

        # Run episode
        for step in range(max_steps):

            # Action
            for a in range(env.get_num_agents()):
                action = agent.act(agent_obs[a], eps=eps)
                action_prob[action] += 1
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            # Build agent specific observations and normalize
            for a in range(env.get_num_agents()):
                data, distance, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                        num_features_per_node=num_features_per_node, current_depth=0)
                data = norm_obs_clip(data, fixed_radius=observation_radius)
                distance = norm_obs_clip(distance)
                agent_data = np.clip(agent_data, -1, 1)
                agent_next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                if done[a]:
                    final_obs[a] = agent_obs[a].copy()
                    final_obs_next[a] = agent_next_obs[a].copy()
                    final_action_dict.update({a: action_dict[a]})
                if not done[a]:
                    agent.step(agent_obs[a], action_dict[a], all_rewards[a], agent_next_obs[a], done[a])
                score += all_rewards[a] / env.get_num_agents()

            # Copy observation
            agent_obs = agent_next_obs.copy()

            if done['__all__']:
                env_done = 1
                for a in range(env.get_num_agents()):
                    agent.step(final_obs[a], final_action_dict[a], all_rewards[a], final_obs_next[a], done[a])
                break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Collection information about training
        done_window.append(env_done)
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))
```

Running the `multi_agent_training.py` file trains a simple agent to navigate to any random target within the railway network. After running you should see a learning curve similiar to this one:

![Learning_Curve](https://i.imgur.com/Po4j4yK.png)

and the agent behavior should look like this:

![Conflict_Avoidence](https://i.imgur.com/AvBHKaD.gif)
