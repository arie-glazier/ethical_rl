import numpy as np

class EpsilonGreedyPolicy:
  def __init__(self, **kwargs):
    self.env = kwargs["environment"]
    self.n_outputs = self.env.action_space.n

  def get_action(self, model, state, epsilon):
      return np.random.randint(self.n_outputs) if np.random.rand() < epsilon else np.argmax(model.predict(state[np.newaxis])[0])