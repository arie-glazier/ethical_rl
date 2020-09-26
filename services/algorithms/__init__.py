from tensorflow import keras
from services.constants import *

class AlgorithmBASE:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]
    self.n_outputs = self.env.action_space.n
    self.model = kwargs[MODEL]
    self.policy = kwargs[POLICY]

    self.batch_size = int(kwargs[BATCH_SIZE])
    self.discount_factor = float(kwargs[DISCOUNT_FACTOR])
    self.loss_function = getattr(keras.losses, kwargs[LOSS_FUNCTION])
    self.learning_rate = float(kwargs[LEARNING_RATE])
    self.optimizer = getattr(keras.optimizers, kwargs[OPTIMIZER])(lr=self.learning_rate, epsilon=1.5e-4) #TODO: this epsilon only works for Adam

    self.number_of_episodes = int(kwargs[NUMBER_OF_EPISODES])
    self.maximum_step_size = int(kwargs[MAXIMUM_STEP_SIZE])