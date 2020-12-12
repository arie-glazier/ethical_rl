# Ethical Reinforcement Learning (ERL)

ERL is a Python library for generating standard RL baselines, training agents with ethical constraints, and evaluating agent performance.  

This repo is implemented on top of [OpenAI gym](https://github.com/openai/gym) and is based heavily on [gym-minigrid](https://github.com/maximecb/gym-minigrid).

## Installation

Clone the repo.  TODO: create import package

## Requirements

```
tensorflow==2.3.0
gym==0.17.2
numpy==1.18.1
gym-minigrid==1.0.1
```

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

results = algorithm.train() # returns a list of total rewards in each episode
```

## Configurations

Adjustable parameters (with examples) that can be specified in config.json or from the command line are:

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
    "policy_module": "ethical_rl.policies.epsilon_greedy",
    "algorithm_module" : "ethical_rl.algorithms.dqn.double_dqn",
    "epsilon_schedule_module" : "ethical_rl.common.schedules.linear",
    "epsilon_start" : 1.0,
    "epsilon_end" : 0.01,
    "epsilon_anneal_percent" : 0.10,
    "environment_wrapper" : {
      "modules" : ["gym_minigrid.wrappers","gym_minigrid.wrappers"], 
      "classes" : ["RGBImgPartialObsWrapper", "ImgObsWrapper"]
    },
    "environment_name" : "MiniGrid-arie-test-v0",
    "replay_buffer_module": "ethical_rl.algorithms.dqn.replay_buffer.simple",
    "replay_buffer_prioritization" : 0.5
  }
```

TODO: explanations of each

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)