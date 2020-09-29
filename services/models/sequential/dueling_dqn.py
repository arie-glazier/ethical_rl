
from ..sequential import SequentialModelBASE
from tensorflow import keras

class Model(SequentialModelBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
    input_states = keras.layers.Input(shape=self.input_shape)
    hidden1 = keras.layers.Dense(self.fully_connected_model_size[0], activation="elu")(input_states) #TODO: allow config of more layers
    hidden2 = keras.layers.Dense(self.fully_connected_model_size[1], activation="elu")(hidden1)
    state_values = keras.layers.Dense(1)(hidden2)
    raw_advantages = keras.layers.Dense(self.n_outputs)(hidden2)
    advantages = raw_advantages - keras.backend.max(raw_advantages, axis=1,keepdims=True)
    Q_values = state_values + advantages

    self.model = keras.Model(
      inputs=[input_states], outputs=[Q_values]
    )