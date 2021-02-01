import argparse, pickle, os, gym, random
import sys
from dataclasses import asdict

from ethical_rl.labels import EpisodeResult, LabeledData
from ethical_rl.labels.synthetic import Labeler as SyntheticLabeler
from ethical_rl.labels.human import Labeler as HumanLabeler

#TODO: save this in pickle file and load dynamically
from ethical_rl.environments.five_by_five import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--pickle_file", default="double_dueling_prioritized_constraint_aware_double_dqn_dueling_dqn_prioritized_5000_MiniGrid-Ethical5x5-v0.pickle")
PARSER.add_argument("--human", action="store_true")
PARSER.add_argument("--data_directory", default=".")

if __name__ == "__main__":
  args = PARSER.parse_args()

  root_directory = os.path.dirname(os.path.abspath(__file__))
  data_directory = args.data_directory or os.path.join(root_directory, "data")
  pickle_file_name = args.pickle_file
  pickle_file_path = os.path.join(data_directory, pickle_file_name)
  with open(pickle_file_path, "rb") as f: data = pickle.load(f)

  data_config = data["config"]
  test_name = data_config["test_name"]
  environment_name = data_config["environment_name"]

  env = gym.make(environment_name)
  env.reset()

  Labeler = HumanLabeler if args.human else SyntheticLabeler
  labeler = Labeler(environment=env)

  data_history = data["history"]
  reward_history = data_history["rewards"]
  constraint_violation_history = data_history["constraint_violation_count"]
  action_history = data_history["episode_action_history"]
  episode_history = [ EpisodeResult(rewards, violations, actions) for rewards, violations, actions in zip(reward_history, constraint_violation_history, action_history)]

  # To generate labels we need to get pairs of episodes for comparison
  print(f"we have {len(episode_history)} episodes")
  episode_pairs = labeler.make_episode_pairs(episode_history)
  labeled_data = labeler.generate(episode_pairs) # TODO: standardize labels between human and synthetic
  print([x.label for x in labeled_data])

  print(f"labeled data has {len(labeled_data)} items")
  # within labeled data, each state for both left and right is a 
  # training observations
  observations_counter = 0
  for item in labeled_data:
    observations_counter += len(item.left_trajectory) + len(item.right_trajectory)
  print(f"this corresponds to {observations_counter} training observations")


  pickle.dump([asdict(x) for x in labeled_data], open("./labeled_data", "wb"))




