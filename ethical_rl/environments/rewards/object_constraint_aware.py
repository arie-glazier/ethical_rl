from ethical_rl.environments.rewards import RewardBASE
from ethical_rl.constants import *

from scipy.spatial import distance
import sys

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.termination_reward = int(kwargs.get(TERMINATION_REWARD, 30))
    self.step_reward = int(kwargs.get(STEP_REWARD, -1))
    self.constraint_violation_penalty = int(kwargs.get("constraint_violation_penalty", -1))
    self.pickup = False
    self.step_counter = 0

  def get(self, **kwargs):
    self.step_counter += 1
    ball_coords = kwargs["ball_coords"]
    distance_to_red = distance.cdist([ball_coords], [(3,1)], "cityblock")[0][0]
    action = kwargs["action"]
    ball_placed_correctly = kwargs["ball_placed_correctly"]

    # if ball_placed_correctly:
    #   if kwargs["done"]: return 1000
    #   else: return -1
    # else:
    #   if kwargs["done"]: return 100/distance_to_red
    #   elif kwargs["was_carrying"] and not kwargs["is_carrying"] and action == 4:
    #     return 100/(1+distance_to_red)
    #   else: return -1 - distance_to_red
    # if action == 4:
    #   print(kwargs["was_carrying"])
    #   print(kwargs["is_carrying"])
    #   print(kwargs["has_ever_been_good"])
    #   print(ball_placed_correctly)


    if kwargs["done"]:
      return 10000 if ball_placed_correctly else 100/distance_to_red
    elif kwargs["was_carrying"] and not kwargs["is_carrying"] and action == 4 and not kwargs["has_ever_been_good"]:
      if ball_placed_correctly:
        return 10000
      else:
        return 1
    #   else: return 100
    elif action == 3 and kwargs["has_ever_been_good"]: #and (not kwargs["is_carrying"] or kwargs["was_carrying"]):
      return -10000
    #   return -20 - distance_to_red**2
    elif action == 4 and not kwargs["was_carrying"]:
      return -10000
    else:
      return  -10 - (10 * distance_to_red**2)