import json, argparse, os, sys, importlib

import gym
from services.environments.minigrid_test import *
from services.util import load_class, load_object, load_model, load_policy, load_algorithm
from services.arguments import Arguments
from services.config import Config
from services.constants import *

pwd = os.path.dirname(os.path.realpath(__file__))
PARSER = Arguments(pwd=pwd).parser 
PARSER.add_argument("--test_name")

if __name__ == "__main__":
  args = PARSER.parse_args()
  config = Config(pwd=pwd, args=args).config

  env = gym.make(config.get(ENVIRONMENT_NAME, args.environment_name), **config)
  environment_wrapper_config = config.get(ENVIRONTMENT_WRAPPER)
  if environment_wrapper_config:
    for idx, module in enumerate(environment_wrapper_config[MODULES]): # for ability to wrap mulitple
      environment_wrapper = load_class(module, environment_wrapper_config[CLASSES][idx])
      env = environment_wrapper(env)
  env.reset()

  model = load_model(config[MODEL_MODULE])(environment=env, **config).model
  policy = load_policy(config[POLICY_MODULE])(environment=env)

  algorithm = load_algorithm(config[ALGORITHM_MODULE])(environment=env, model=model, policy=policy, **config)
  results, losses = algorithm.train()

  # can delete, just for testing
  # print(results[-20:])
  if results:
    import matplotlib.pyplot as plt
    plt.clf() # render messes this up
    plt.plot(range(0, len(results)), results)
    plt.xlabel("episodes")
    plt.ylabel("total reward")
    algorithm_name = config[ALGORITHM_MODULE].split(".")[-1]
    model_name = config[MODEL_MODULE].split(".")[-1]
    replay_name = config[REPLAY_BUFFER_MODULE].split(".")[-1]
    title = f"{algorithm_name}_{model_name}_{replay_name}_{config[NUMBER_OF_EPISODES]}"
    plt.title(title)
    print(f"title is {title}")
    plt.savefig(f"./results/{title}.png")

    plt.clf()
    plt.plot(range(0, len(losses)), losses)
    plt.xlabel("episodes")
    plt.ylabel("total loss")
    plt.show()
