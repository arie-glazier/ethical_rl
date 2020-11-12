import sys

from services.constants import *
from services.labels import LabelerBASE, LabeledData

class Labeler(LabelerBASE):

  def generate(self, episode_pairs):
    # From "Learning from Humans" we have 4 possible outcomes:
    #   + Left is better => "left"
    #   + Right is better => "right"
    #   + They are the same => "same"
    #   + Can't tell => "unknown"
    labeled_data = []
    for left, right in episode_pairs:
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
        label = "right"
      else:
        print("We shouldnt ever get here")
        print(left, right)
        sys.exit()

      # we can only train on results where we have a label, so only need to (re)generate states in those cases
      # TODO: should we save state history during training? will be a lot of space
      # TODO: will this work for training in random start states?  reason to track state in training
      if label == "left" or label == "right" or label == "same":
        left_trajectory = self.get_trajectory(left)
        right_trajectory = self.get_trajectory(right)

        labeled_data.append(LabeledData(label=label, left_trajectory=left_trajectory, right_trajectory=right_trajectory))

    return labeled_data
