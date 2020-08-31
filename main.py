import json, argparse, os

import gym

from services.environments.maker import EnvironmentMaker
from services.models.sequential import SequentialModel
from services.policies.epsilon_greedy import EpsilonGreedyPolicy
from services.algorithms.dqn import DQN

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--test_name")

if __name__ == "__main__":
  args = PARSER.parse_args()
  pwd = os.path.dirname(os.path.realpath(__file__))

  config = json.loads(open(os.path.join(pwd, "config.json")).read())[args.test_name]

  env = gym.make(config["environment_name"]) 
  model = SequentialModel(environment=env).simple_model()
  policy = EpsilonGreedyPolicy(environment=env, model=model)

  algorithm = DQN(environment=env, model=model, policy=policy, **config)
  algorithm.train()