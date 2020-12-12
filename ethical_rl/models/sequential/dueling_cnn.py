import sys
from ethical_rl.models.sequential import SequentialModelBASE
from tensorflow import keras

class Model(SequentialModelBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    input_states = keras.layers.Input(shape=self.input_shape)
    h1 = keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape)(input_states)
    h2 = keras.layers.MaxPooling2D((2, 2))(h1)
    h3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(h2)
    h4 = keras.layers.MaxPooling2D((2, 2))(h3)
    h5 = keras.layers.Conv2D(64, (3, 3), activation='relu')(h4)
    h6 = keras.layers.Flatten()(h5)

    state_values = keras.layers.Dense(1)(h6)
    raw_advantages = keras.layers.Dense(self.n_outputs)(h6)
    advantages = raw_advantages - keras.backend.max(raw_advantages, axis=1,keepdims=True)
    Q_values = state_values + advantages
    self.model = keras.Model(
      inputs=[input_states], outputs=[Q_values]
    )