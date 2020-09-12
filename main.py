import json, argparse, os, sys

import gym
from services.environments.minigrid_test import *

from services.environments.maker import EnvironmentMaker
from services.models.sequential import SequentialModel
from services.policies.epsilon_greedy import EpsilonGreedyPolicy
from services.algorithms.dqn import DQN

from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, FlatObsWrapper

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--test_name")
PARSER.add_argument("--environment_name")

if __name__ == "__main__":
  args = PARSER.parse_args()
  pwd = os.path.dirname(os.path.realpath(__file__))

  full_config = json.loads(open(os.path.join(pwd, "config.json")).read())
  default_config = full_config["DEFAULT_HYPERPARAMETERS"]
  test_config = full_config.get(args.test_name, {})
  config = {**default_config, **test_config}
  
  env = FlatObsWrapper(gym.make(config.get("environment_name", args.environment_name))) # this flattens rgb representationo
  env.reset()

  model = SequentialModel(environment=env).simple_model()
  policy = EpsilonGreedyPolicy(environment=env, model=model)

  algorithm = DQN(environment=env, model=model, policy=policy, **config)
  algorithm.train()