from ethical_rl.constants import *

class RewardBASE:
  def __init__(self, **kwargs):
    self.environment = kwargs[ENVIRONMENT]
