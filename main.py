import json, argparse, os, sys, importlib

import gym
from services.environments.minigrid_test import *
from services.util import load_class, load_object
from services.arguments import Arguments
from services.config import Config

import matplotlib.pyplot as plt

pwd = os.path.dirname(os.path.realpath(__file__))
PARSER = Arguments(pwd=pwd).parser 
PARSER.add_argument("--test_name")

if __name__ == "__main__":
  args = PARSER.parse_args()
  config = Config(pwd=pwd, args=args).config

  env = gym.make(config.get("environment_name", args.environment_name))
  environment_wrapper_config = config.get("environment_wrapper")
  if environment_wrapper_config:
    for idx, module in enumerate(environment_wrapper_config["modules"]): # for ability to wrap mulitple
      environment_wrapper = load_class(module, environment_wrapper_config["classes"][idx])
      env = environment_wrapper(env)
  env.reset()

  model = getattr(load_object(config["model_type"])(environment=env), config["model_type"]["method"])()
  policy = load_object(config["policy_type"])(environment=env)

  algorithm = load_object(config["algorithm_type"])(environment=env, model=model, policy=policy, **config)
  results = algorithm.train()

  # can delete, just for testing
  plt.plot(range(0, len(results)), results)
  plt.savefig("results.png")