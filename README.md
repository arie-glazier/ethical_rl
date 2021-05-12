# Ethical Reinforcement Learning (ERL)

ERL is a Python library for generating standard RL baselines, training agents with ethical constraints, and evaluating agent performance.  

This repo is implemented on top of [OpenAI gym](https://github.com/openai/gym) and is based heavily on [gym-minigrid](https://github.com/maximecb/gym-minigrid).

## Installation

This repo is available as a PyPi package and can be installed using ```pip install ethical-rl```

However, I make no guarantees that the package on PyPi is up to date so if you want the code you see here: Clone the repo and ```pip install .```

## Requirements

Library dependencies are listed in setup.py.

NOTE: The gym-minigrid package may not be up to date in the pip package manager.  It may be necessary to clone the gym-minigrid repo, navigate into the directory, and ```pip install .```.

## Usage

This library can be used in two ways.

### 1.) Configuration Driven

Create an entry in ```config.json``` of the form that has an arbitrary test name as the key and values corresponding to the test to run.  Any value that is not specified is defaulted from values in ```default_config.json```.

A test can be run by:
```bash
python main.py --test_name your_test_name
```

If at run time, you wish to override an argument from ```config.json```, simply pass it as a command line argument:

```bash
python main.py --test_name your_test_name --number_of_episodes 1000
```

For a description of available configurations, see the "Configurations" section below.

### 2.) Roll Your Own

The main elements of a RL problem are an environment, a model, a policy, and an algorithm.  These items, along with their necessary configuration parameters (see "Configurations" below), are all that are needed to train an agent.


```python
import json, gym

from ethical_rl.models.sequential.perceptron import Model
from ethical_rl.policies.epsilon_greedy import Policy
from ethical_rl.algorithms.dqn.double_dqn import Algorithm

config = json.loads(open("./default_config.json").read())

environment = gym.make("CartPole-v0")
model = Model(environment=environment).model
policy = Policy(environment=environment)
algorithm = Algorithm(environment=environment, model=model, policy=policy, **config)

algorithm.train() 
```

## Configurations

Adjustable parameters defaults that can be specified in config.json or from the command line are:

```json
  {
     "max_replay_buffer_length" : 2000,
    "batch_size" : 32,
    "discount_factor" : 0.95,
    "optimizer" : "Adam",
    "learning_rate" : 1e-3,
    "loss_function" : "mean_squared_error",
    "number_of_episodes" : 600,
    "maximum_step_size" : 200,
    "buffer_wait_steps" : 50,
    "model_module" : "ethical_rl.models.sequential.perceptron",
    "fully_connected_model_size" : [32, 32],
    "policy_module": "ethical_rl.policies.epsilon_greedy",
    "algorithm_module" : "ethical_rl.algorithms.dqn.double_dqn",
    "reward_module" : "ethical_rl.environments.rewards.negative_step",
    "termination_reward" : 30,
    "step_reward" : -1,
    "epsilon_schedule_module" : "ethical_rl.common.schedules.linear",
    "epsilon_start" : 1.0,
    "epsilon_end" : 0.01,
    "epsilon_anneal_percent" : 0.10,
    "environment_wrapper" : {
      "modules" : ["ethical_rl.wrappers.symbolic_observations"], 
      "classes" : ["SymbolicObservationsOneHotWrapper"]
    },
    "environment_name" : "MiniGrid-Ethical5x5-v0",
    "replay_buffer_module": "ethical_rl.algorithms.dqn.replay_buffer.simple",
    "replay_buffer_prioritization" : 0.5,
    "target_sync_frequency" : 50,
    "clip_norm": null,
    "max_steps_per_episode" : 100,
    "render_training_steps" : null,
    "random_start_position" : false,
    "constraint_violation_penalty": -1,
    "include_environment_config": true,
    "reward_model_path" : "",
    "results_destination": "",
    "constraint_color": null,
    "constraint_location": null,
    "clip_ratio": 0.0,
    "rollout_length": 128,
    "target_kl": null,
    "evaluate_steps": 25,
    "td_lambda_value": 1,
    "loss_steps": null,
    "render_steps": null,
    "num_epochs": 1,
    "alpha": 1.0,
    "initial_lambda_value": 1.0,
    "lambda_learning_rate": 0.001,
    "classifier_training_steps": 1,
    "policy_training_steps": 1
```

## Examples

Boilerplate code for common uses is available in the ```examples``` folder.
* ```view_episode.py``` - visual rendering of a single episode
* ```generate_labels.py``` - generate labels from human/synthetic feedback to be used in reward function approximation
* ```train_reward_predictor.py``` - create a neural network for reward function approximation
* ```deploy.py``` - deploy code and configuration to run on a remote host

## Repo Organization

As noted above, the main components of an RL problem are an environment, a model, a policy, and an algorithm.  This repository is organized in such a way that each of these components have a common interface and can be easily modified or created and used in whatever combination is desired.

### Environments

Available environments can be found in ```./environments```.  Environments must inherit from ```gym.Env``` and required override methods are: ```reset()``` and ```step()```.  For more information see the [OpenAI Gym environment docs](https://github.com/openai/gym/blob/master/docs/creating-environments.md)

Currently, we have 5 environments - 4 different grid worlds and 1 for news article recommendation.

![Alt text](/static/8x8.png?raw=true "8x8")

To create a new environment, simply create a new file with an ```Environment``` class, register it with a unique ID, and pass the ID in via the ```environment_name``` argument.

### Wrappers

OpenAI Gym supports the concept of "wrappers".  Wrappers allow environment transformations to be done in a modular fashion.  For example, when states are represented as RGB images, often pixels are represented by values between 0 and 255.  However, sometimes learning can be done faster if pixel values are rescaled to take values between 0 and 1.  This is where a wrapper can be used.  There are several wrappers provided by OpenAI Gym (see: ).  Also, [this write-up](https://alexandervandekleut.github.io/gym-wrappers/) provides an in depth explanation of wrappers and how to create your own.

Custom wrappers in this repo can be placed in ```./wrappers``` and passed in via the ```environment_wrapper``` configuration where you reference the module path and the class name.

### Rewards

A key component of an RL environment is a reward function.  In our tests here we are concerned with evaluating agent behavior when constraints are explicitly modeled in the reward functions against behavior when constraint information is provided by other means such as labeled trajectories or demonstrations.

Custom ```Reward``` classes can be created by adding a new file in ```./environments/rewards``` and then referencing the new object with the ```reward_module``` path.

### Algorithms, Models, and Policies

These items all follow the exact same pattern as rewards.  To create a new implementation, simply add a file to either ```./algorithms```, ```./models```, or ```./policies``` with a class name of ```Algorithm```, ```Model```, or ```Policy```.  Then, reference your new object with the appropriate configuration argument (either ```algorithm_module```, ```model_module```, or ```policy_module```).

Each of these has an associated ```BASE``` class that can be inherited from to streamline common attribute instantiation (e.g. ```batch_size``` or ```learning_rate``` for models) that helps keep new class files relatively light weight.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License
[MIT](https://choosealicense.com/licenses/mit/)