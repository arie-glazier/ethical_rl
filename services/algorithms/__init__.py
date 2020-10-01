from tensorflow import keras
from services.constants import *
from services.util import load_schedule

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
    self.clip_norm = float(kwargs[CLIP_NORM])
    # Does not perform clipping unless clip_norm is set to non-zero float
    if self.clip_norm == 0: self.clip_norm = None

    self.number_of_episodes = int(kwargs[NUMBER_OF_EPISODES])
    self.maximum_step_size = int(kwargs[MAXIMUM_STEP_SIZE])

    self.epsilon_start = kwargs[EPSILON_START]
    self.epsilon_end = kwargs[EPSILON_END]
    self.epsilon_anneal_percent = kwargs[EPSILON_ANNEAL_PERCENT]
    self.epsilon_schedule_timesteps = int(self.number_of_episodes * self.epsilon_anneal_percent)
    self.epsilon_schedule = load_schedule(kwargs[EPSILON_SCHEDULE_MODULE])(self.epsilon_schedule_timesteps, self.epsilon_end, self.epsilon_start)