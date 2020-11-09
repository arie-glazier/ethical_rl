import argparse, pickle, os, gym, random
from dataclasses import dataclass, asdict
from typing import List
import sys

#TODO: save this in pickle file and load dynamically
from services.environments.five_by_five import *

@dataclass
class EpisodeResult:
  total_reward: int
  total_constraint_violations: int
  actions: List[int]

@dataclass
class LabeledData:
  label: str
  left_state: List[dict]
  right_state: List[dict]

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--pickle_file")

if __name__ == "__main__":
  args = PARSER.parse_args()

  root_directory = os.path.dirname(os.path.abspath(__file__))
  data_directory = os.path.join(root_directory, "data")
  pickle_file_name = "double_dueling_prioritized_constraint_aware_double_dqn_dueling_dqn_prioritized_5000_MiniGrid-Ethical5x5-v0.pickle" # args.pickle_file
  pickle_file_path = os.path.join(data_directory, pickle_file_name)

  with open(pickle_file_path, "rb") as f: data = pickle.load(f)


  data_history = data["history"]
  reward_history = data_history["rewards"]
  constraint_violation_history = data_history["constraint_violation_count"]
  action_history = data_history["episode_action_history"]
  episode_history = [ EpisodeResult(rewards, violations, actions) for rewards, violations, actions in zip(reward_history, constraint_violation_history, action_history)]

  # To generate labels we need to get pairs of episodes for comparison
  print(f"we have {len(episode_history)} episodes")
  # Randomly shuffle episodes
  random.shuffle(episode_history)
  # Split list in half (TODO: should check to make sure we have an even number of episdoes)
  episodes_left = episode_history[:len(episode_history) // 2]
  episodes_right = episode_history[len(episode_history) // 2:]
  # Create pairs.  Result is an iterator where each item is (episode_result_a, episode_result_b)
  episode_pairs = zip(episodes_left, episodes_right)

  data_config = data["config"]
  test_name = data_config["test_name"]
  environment_name = data_config["environment_name"]

  env = gym.make(environment_name)
  env.reset()

  # From the "Learning from Humans" we have 4 possible outcomes:
  #   + Left is better => "left"
  #   + Right is better => "right"
  #   + They are the same => "same"
  #   + Can't tell => "unknown"
  labeled_data = []
  c = 0
  for left, right in episode_pairs:
    c += 1
    if c % 10 == 0: print(f"pair {c}")
    label = None
    # same reward and same number of violations
    if left.total_reward == right.total_reward and left.total_constraint_violations == right.total_constraint_violations:
      label = "same"
    # more reward but also more violations
    elif left.total_reward >= right.total_reward and left.total_constraint_violations >= right.total_constraint_violations:
      label = "unknown"
    # more reward but also more violations
    elif right.total_reward >= left.total_reward and right.total_constraint_violations >= left.total_constraint_violations:
      label = "unknown"
    # left is better
    elif left.total_reward >= right.total_reward and left.total_constraint_violations <= right.total_constraint_violations:
      label = "left"
    # right is better
    elif right.total_reward >= left.total_reward and right.total_constraint_violations <= left.total_constraint_violations:
      label = "left"
    else:
      print("We shouldnt ever get here")
      print(left, right)
      sys.exit()

    # we can only train on results where we have a label, so only need to (re)generate states in those cases
    # TODO: should we save state history during training? will be a lot of space
    if label == "left" or label == "right":
      left_start_state = env.reset() # this won't work for training in random start states.  reason to track state in training
      left_state = [left_start_state] + [ env.step(action) for action in left.actions ]
      right_start_state = env.reset() # this won't work for training in random start states.  reason to track state in training
      right_state = [right_start_state] + [ env.step(action) for action in right.actions ]
      labeled_data.append(LabeledData(label=label, left_state=left_state, right_state=right_state))

  print(f"labeled data has {len(labeled_data)} items")
  # within labeled data, each state for both left and right is a 
  # training observations
  observations_counter = 0
  for item in labeled_data:
    observations_counter += len(item.left_state) + len(item.right_state)
  print(f"this corresponds to {observations_counter} training observations")

  pickle.dump([asdict(x) for x in labeled_data], open("./labeled_data", "wb"))




