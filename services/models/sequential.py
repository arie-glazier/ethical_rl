import sys
from tensorflow import keras

class SequentialModel:
  def __init__(self, **kwargs):
    self.env = kwargs["environment"]

    # this is the structure of obs. space https://github.com/openai/gym/issues/593
    # EX: cartpole = [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    self.input_shape = self.env.observation_space.shape

    self.n_outputs = self.env.action_space.n

  def simple_model(self):
    return keras.models.Sequential([
      keras.layers.Dense(32, activation="elu", input_shape=self.input_shape),
      keras.layers.Dense(32, activation="elu"),
      keras.layers.Dense(self.n_outputs)
    ])

  def simple_model_single(self):
    return keras.models.Sequential([
      keras.layers.Dense(32, activation="elu", input_shape=self.input_shape),
      keras.layers.Dense(self.n_outputs)
    ])
  
  def dueling_dqn(self):
    input_states = keras.layers.Input(shape=self.input_shape)
    hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
    hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
    state_values = keras.layers.Dense(1)(hidden2)
    raw_advantages = keras.layers.Dense(self.n_outputs)(hidden2)
    advantages = raw_advantages - keras.backend.max(raw_advantages, axis=1,keepdims=True)
    Q_values = state_values + advantages
    return keras.Model(
      inputs=[input_states], outputs=[Q_values]
    )

  def conv(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(self.n_outputs))
    return model

  def dueling_conv(self):
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
    return keras.Model(
      inputs=[input_states], outputs=[Q_values]
    )