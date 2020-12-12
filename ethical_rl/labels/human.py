import time

from ethical_rl.constants import *
from ethical_rl.labels import LabelerBASE, LabeledData

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

      time.sleep(1.5)
      left_trajectory = self.get_trajectory(left, display=True)

      self.environment.render(mode="close") # i think this clears the screen?

      time.sleep(1.5)
      right_trajectory = self.get_trajectory(right, display=True)

      # TODO: check valid input, better messaging, option to quit early, make dataclass for this
      label = input("1 - 4? ")
      
      labeled_data.append(LabeledData(label=label, left_trajectory=left_trajectory, right_trajectory=right_trajectory))
      if len(labeled_data) == 100 or label == "q": break #TODO: add configuration for number of items

    return labeled_data