# How to train an Agent on Flatland
Quick introduction on how to train a simple DQN agent using Flatland and Pytorch. At the end of this Tutorial you should be able to train a single agent to navigate in Flatland.
We use the `training_navigation.py` ([here](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/training_navigation.py)) file to train a simple agent with the tree observation to solve the navigation task.

## Actions in Flatland
Flatland is a railway simulation. Thus the actions of an agent are strongly limited to the railway network. This means that in many cases not all actions are valid.
The possible actions of an agent are

- 0 *Do Nothing*:  If the agent is moving it continues moving, if it is stopped it stays stopped
- 1 *Deviate Left*: This action is only valid at cells where the agent can change direction towards left. If action is chosen, the left transition and a rotation of the agent orientation to the left is executed. If the agent is stopped at any position, this action will cause it to start moving in any cell where forward or left is allowed!
- 2 *Go Forward*: This action will start the agent when stopped. At switches this will chose the forward direction.
- 3 *Deviate Right*: Exactly the same as deviate left but for right turns.
- 4 *Stop*: This action causes the agent to stop, this is necessary to avoid conflicts in multi agent setups (Not needed for navigation).

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
Before you get started with the training make sure that you have [pytorch](https://pytorch.org/get-started/locally/) installed.
Let us now train a simPle double dueling DQN agent to navigate to its target on flatland. We start by importing flatland

```
from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import norm_obs_clip, split_tree
```

For this simple example we want to train on randomly generated levels using the `complex_rail_generator`. We use the following parameter for our first experiment:

```
# Parameters for the Environment
x_dim = 10
y_dim = 10
n_agents = 1
n_goals = 5
min_dist = 5
```

As mentioned above, for this experiment we are going to use the tree observation and thus we load the observation builder:

```
# We are training an Agent using the Tree Observation with depth 2
observation_builder = TreeObsForRailEnv(max_depth=2)
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

```
env_renderer = RenderTool(env, gl="PILSVG", )
```

### Setting up the agent

To set up a appropriate agent we need the state and action space sizes. From the discussion above about the tree observation we end up with:

[**Adrian**: I just wonder, why this is not done in seperate method in the the observation: get_state_size, then we don't have to write down much more. And the user don't need to 
understand anything about the observation. I suggest moving this into the observation, base ObservationBuilder declare it as an abstract method. ... ] 

```
# Given the depth of the tree observation and the number of features per node we get the following state_size
features_per_node = 9
tree_depth = 2
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = features_per_node * nr_nodes

# The action space of flatland is 5 discrete actions
action_size = 5
```

In the `training_navigation.py` file you will find further variable that we initiate in order to keep track of the training progress.
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
[**Adrian**: Same question as above, why not done in the observation class?]

```
for trials in range(1, n_trials + 1):

    # Reset environment
    obs, info = env.reset(True, True)
    if not Training:
        env_renderer.set_new_rail()

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

    # Reset score and done
    score = 0
    env_done = 0

    # Run episode
    for step in range(max_steps):

        # Only render when not triaing
        if not Training:
            env_renderer.renderEnv(show=True, show_observations=True)

        # Chose the actions
        for a in range(env.get_num_agents()):
            if not Training:
                eps = 0

            action = agent.act(agent_obs[a], eps=eps)
            action_dict.update({a: action})

            # Count number of actions takes for statistics
            action_prob[action] += 1

        # Environment step
        next_obs, all_rewards, done, _ = env.step(action_dict)

        for a in range(env.get_num_agents()):
            rail_data, distance_data, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                              num_features_per_node=features_per_node,
                                                              current_depth=0)
            rail_data = norm_obs_clip(rail_data)
            distance_data = norm_obs_clip(distance_data)
            agent_data = np.clip(agent_data, -1, 1)
            agent_next_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))

        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):

            # Remember and train agent
            if Training:
                agent.step(agent_obs[a], action_dict[a], all_rewards[a], agent_next_obs[a], done[a])

            # Update the current score
            score += all_rewards[a] / env.get_num_agents()

        agent_obs = agent_next_obs.copy()
        if done['__all__']:
            env_done = 1
            break

    # Epsilon decay
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon
```

Running the `training_navigation.py` file trains a simple agent to navigate to any random target within the railway network. After running you should see a learning curve similiar to this one:

![Learning_curve](https://i.imgur.com/yVGXpUy.png)

and the agent behavior should look like this:

![Single_Agent_Navigation](https://i.imgur.com/t5ULr4L.gif)

