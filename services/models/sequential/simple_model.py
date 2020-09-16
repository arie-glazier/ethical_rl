import sys
from ..sequential import SequentialModelBASE
from tensorflow import keras

class Model(SequentialModelBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.model = keras.models.Sequential([
      keras.layers.Dense(32, activation="elu", input_shape=self.input_shape),
      keras.layers.Dense(32, activation="elu"),
      keras.layers.Dense(self.n_outputs)
    ])
