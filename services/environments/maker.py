import gym

class EnvironmentMaker:
  def __init__(self, **kwargs):
    self.env = gym.make(kwargs["environment_name"])
    self.input_shape = self.env.observation_space.shape
    self.n_outputs = self.env.action_space.n
  
  def get_env(self):
    return self.env