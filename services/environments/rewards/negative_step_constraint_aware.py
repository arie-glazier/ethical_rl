from services.environments.rewards import RewardBASE
from services.constants import *

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.termination_reward = int(kwargs.get(TERMINATION_REWARD, 30))
    self.step_reward = int(kwargs.get(STEP_REWARD, -1))
    self.constraint_violation_penalty = int(kwargs.get("constraint_violation_penalty", -1))

  def get(self, done, constraint_violation=False):
    if done:
      return self.termination_reward
    return self.step_reward + (self.constraint_violation_penalty if constraint_violation else 0)