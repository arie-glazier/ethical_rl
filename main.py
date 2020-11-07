import json, argparse, os, sys, importlib

import gym
# TODO: make these imports dynamic
from services.environments.minigrid_test import *
from services.environments.five_by_five import *
from services.environments.ten_by_ten import *
from services.util import load_class, load_object, load_model, load_policy, load_algorithm
from services.arguments import Arguments
from services.config import Config
from services.constants import *

pwd = os.path.dirname(os.path.realpath(__file__))

PARSER = Arguments(pwd=pwd).parser 
PARSER.add_argument("--test_name")
PARSER.add_argument("--server_execution", action="store_true")

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
  history = algorithm.train()

  # can delete, just for testing
  # print(results[-20:])
  def make_plot(plt, x_label, y_label, title, data):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(range(0, len(data)), data)
    return plt

  if history:
    import pickle

    algorithm_name = config[ALGORITHM_MODULE].split(".")[-1]
    model_name = config[MODEL_MODULE].split(".")[-1]
    replay_name = config[REPLAY_BUFFER_MODULE].split(".")[-1]
    environment_name = config[ENVIRONMENT_NAME]
    title = f"{config[TEST_NAME]}_{algorithm_name}_{model_name}_{replay_name}_{config[NUMBER_OF_EPISODES]}_{environment_name}"
    print(f"title is {title}")

    data = {
      CONFIG:config,
      HISTORY: history
    }

    pickle.dump(data, open(f"./data/{title}.pickle", "wb"))

    if not args.server_execution:
      import matplotlib.pyplot as plt
      plt.clf() # render messes this up
      plt = make_plot(plt, "episodes", "total_reward", title, history[REWARDS])
      plt.savefig(f"./results/{title}.png")
      plt.show()

      plt.clf()
      constraint_title = f"{title}_CONSTRAINTS"
      plt = make_plot(plt, "episodes", "total_violations", constraint_title, history[CONSTRAINT_VIOLATION_COUNT])
      plt.savefig(f"./results/{constraint_title}.png")
      plt.show()

      plt.clf()
      plt = make_plot(plt, "episodes", "total_loss", f"{title}_LOSS", history[LOSS])
      plt.show()