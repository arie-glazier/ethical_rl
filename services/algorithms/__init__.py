from tensorflow import keras
from services.constants import *

class AlgorithmBASE:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]
    self.n_outputs = self.env.action_space.n
    self.model = kwargs[MODEL]
    self.policy = kwargs[POLICY]

    self.batch_size = kwargs[BATCH_SIZE]
    self.discount_factor = kwargs[DISCOUNT_FACTOR]
    self.loss_function = getattr(keras.losses, kwargs[LOSS_FUNCTION])
    self.optimizer = getattr(keras.optimizers, kwargs[OPTIMIZER])(lr=kwargs[LEARNING_RATE])

    self.number_of_episodes = int(kwargs[NUMBER_OF_EPISODES])
    self.maximum_step_size = kwargs[MAXIMUM_STEP_SIZE]