from ethical_rl.reporting import Reporter
import pickle, os, sys, gym
from scipy import interpolate
from scipy.interpolate import interp1d
import pandas as pd
import argparse
from ethical_rl.environments.five_by_five import *
from ethical_rl.util import load_class, load_object, load_model, load_policy, load_algorithm

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--pickle_file_folder")

if __name__ == "__main__":
  args = PARSER.parse_args()
  pickle_file_folder = args.pickle_file_folder
  # print(list(os.walk(pickle_file_folder)))
  # sys.exit()
  pickle_files = [f"{pickle_file_folder}/{x}" for x in list(os.walk(pickle_file_folder))[0][2] if ".pickle" in x]

  results_destination = "."

  reward_x_label = "episode"
  constraint_x_label = "episode"
  reward_y_label = "total_rewards"
  constraint_y_label = "constraint_violations"
  reward_title = "total rewards by episode"
  constraint_title = "total constraint violations by episode"


  data_list = []
  for f in pickle_files:
    with open(f,"rb") as f: data = pickle.load(f)
    data_list.append(data)

  env = gym.make('MiniGrid-Ethical5x5-v0', reward_module="ethical_rl.environments.rewards.negative_step", termination_reward=1)
  environment_wrapper_config = {
      "modules" : ["ethical_rl.wrappers.symbolic_observations"], 
      "classes" : ["SymbolicObservationsOneHotWrapper"]
  }
  if environment_wrapper_config:
    for idx, module in enumerate(environment_wrapper_config[MODULES]): # for ability to wrap mulitple
      environment_wrapper = load_class(module, environment_wrapper_config[CLASSES][idx])
      env = environment_wrapper(env)
  env.reset()

  true_rewards = {}
  for data in data_list:
    true_rewards[data['config']['reward_model_path']] = []
    episode_history = data['history']['episode_action_history']
    env.reset()
    for episode in episode_history:
      env.reset()
      total_rewards = 0
      for action in episode:
        next_state, reward, done, info= env.step(action)
        total_rewards += reward
      true_rewards[data['config']['reward_model_path']].append(total_rewards)

  reporter = Reporter(results_destination=results_destination)

  reporter.make_stacked_graph(
    [(name,pd.DataFrame(x).rolling(window=100).mean().values.tolist()) for name, x in true_rewards.items()],
    reward_x_label,
    reward_y_label,
    reward_title,
    "lower right"
  )
  # reporter.show_graph()
  reporter.save_figure("true_rewards")
  sys.exit()

  # reward graph
  reporter.make_stacked_graph(
    [(x['config']['reward_model_path'],pd.DataFrame(x['history']['rewards']).rolling(window=50).mean().values.tolist()) for x in data_list],
    reward_x_label,
    reward_y_label,
    reward_title,
    "lower right"
  )
  # reporter.show_graph()
  reporter.save_figure("rewards")

  # constraint graph
  reporter.make_stacked_graph(
    [(x['config']['reward_model_path'],pd.DataFrame(x['history']['constraint_violation_count']).rolling(window=50).mean().values.tolist()) for x in data_list],
    constraint_x_label,
    constraint_y_label,
    constraint_title,
    "upper left"
  )
  # reporter.show_graph()
  reporter.save_figure("constraints")

