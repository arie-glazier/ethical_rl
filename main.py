import json, argparse, os, sys, importlib, datetime, uuid

import gym
# TODO: make these imports dynamic
from ethical_rl.environments.minigrid_test import *
from ethical_rl.environments.five_by_five import *
from ethical_rl.environments.five_by_five_object import *
from ethical_rl.environments.ten_by_ten import *
from ethical_rl.util import load_class, load_object, load_model, load_policy, load_algorithm
from ethical_rl.arguments import Arguments
from ethical_rl.config import Config
from ethical_rl.constants import *
from ethical_rl.reporting import Reporter

import tensorflow as tf
tf.keras.backend.set_floatx('float32')

pwd = os.path.dirname(os.path.realpath(__file__))

PARSER = Arguments(pwd=pwd).parser 
PARSER.add_argument("--test_name")
PARSER.add_argument("--server_execution", action="store_true")
PARSER.add_argument("--result_save_folder")
PARSER.add_argument("--experiment_group_name", default="")

if __name__ == "__main__":
  start = datetime.datetime.now()
  print(start)
  args = PARSER.parse_args()
  config = Config(pwd=pwd, args=args).config

  # TODO: move this out of main driver
  experiment_group_name = args.experiment_group_name or args.test_name # for running multiple iterations
  experiment_identifier = str(uuid.uuid4())
  result_save_folder = os.path.join(args.result_save_folder or "./results", experiment_group_name, experiment_identifier)
  data_save_folder = os.path.join(result_save_folder, "data") or "./data"
  graph_save_folder = os.path.join(result_save_folder, "graphs") or "./graphs"
  if not os.path.exists(result_save_folder): os.makedirs(result_save_folder)
  if not os.path.exists(data_save_folder): os.makedirs(data_save_folder)
  if not os.path.exists(graph_save_folder): os.makedirs(graph_save_folder)

  environment_config = config if config["include_environment_config"] else {}
  env = gym.make(config.get(ENVIRONMENT_NAME, args.environment_name), **environment_config)
  environment_wrapper_config = config.get(ENVIRONTMENT_WRAPPER)
  if environment_wrapper_config:
    for idx, module in enumerate(environment_wrapper_config[MODULES]): # for ability to wrap mulitple
      environment_wrapper = load_class(module, environment_wrapper_config[CLASSES][idx])
      env = environment_wrapper(env)
  env.reset()

  model = load_model(config[MODEL_MODULE])(environment=env, **config).model
  policy = load_policy(config[POLICY_MODULE])(environment=env, **config)

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

    pickle.dump(data, open(f"{data_save_folder}/{title}.pickle", "wb"))

    print(datetime.datetime.now() - start)

    if not args.server_execution:
      # TODO: fix bugs in this
      # reporter = Reporter(history)
      # plt = reporter.create_graph(REWARDS, "episode", "total_reward", title, title)
      # constraint_title = f"{title}_CONSTRAINTS"
      # plt = reporter.create_graph(CONSTRAINT_VIOLATION_COUNT, "episode", "total_violations", constraint_title, constraint_title)
      import matplotlib.pyplot as plt
      plt.clf() # render messes this up
      plt = make_plot(plt, "episodes", "total_reward", title, history[REWARDS])
      plt.savefig(f"{result_save_folder}/{title}.png")
      plt.show()

      plt.clf()
      constraint_title = f"{title}_CONSTRAINTS"
      plt = make_plot(plt, "episodes", "total_violations", constraint_title, history[CONSTRAINT_VIOLATION_COUNT])
      plt.savefig(f"{result_save_folder}/{constraint_title}.png")
      plt.show()

      plt.clf()
      plt = make_plot(plt, "episodes", "total_loss", f"{title}_LOSS", history[LOSS])
      plt.show()
    