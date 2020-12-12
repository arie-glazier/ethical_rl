import sys
from ethical_rl.models.sequential import SequentialModelBASE
from tensorflow import keras

# To make this the safety gridworlds perceptron, have 2 layers of 100 nodes each
class Model(SequentialModelBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    input_layer = keras.layers.Dense(self.fully_connected_model_size[0], activation="elu", input_shape=self.input_shape)
    middle_layers = [keras.layers.Dense(x, activation="elu") for x in self.fully_connected_model_size[1:]] if self.fully_connected_model_size else []

    self.model = keras.models.Sequential([
      input_layer,
      *middle_layers,
      keras.layers.Dense(self.n_outputs)
    ])