from services.environments.rewards import RewardBASE

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def get(self, *args):
    return 1 - 0.9 * (self.environment.step_count / self.environment.max_steps)