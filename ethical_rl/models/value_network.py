from tf_agents.networks.value_network import ValueNetwork
from ethical_rl.constants import *
import sys
import tensorflow_constrained_optimization as tfco
import tensorflow as tf

# this is the value network for PPO
class CustomValueNetwork(ValueNetwork):
  def __init__(self, environment_spec, fc_layer_params, classifier, lambda_var):
    super().__init__(environment_spec, fc_layer_params=fc_layer_params)
    self.classifier = classifier

    lambda_var = tf.Variable([100.0])
    lambda_value = lambda_var.numpy() if lambda_var else [1.0]
    self.lambda_var = tf.constant(lambda_value, name="lambda")

    # TODO: make configurable
    self.alpha = tf.constant(1.0, name="alpha")

  def call(self, observations, step_type=(), network_state=(), **model_kwargs):
    nominal_value = super().call(observations, step_type, network_state)
    value = nominal_value[0]
    if self.classifier:
      predictions = self.classifier(observations)
      cost_function = tf.constant(1.0) - predictions
      penalty = self.lambda_var * (tf.reshape(cost_function, tf.shape(nominal_value[0]).numpy()) - self.alpha)
      value = value - penalty

    return (value, nominal_value[1])

class Model:
  def __init__(self, **kwargs):
    self.environment = kwargs[ENVIRONMENT]

    self.classifier = kwargs.get("classifier")
    self.model_size = kwargs[FULLY_CONNECTED_MODEL_SIZE]
    self.lambda_var = kwargs.get("lambda_var")
    lambda_var = tf.Variable([100.0])
    lambda_value = lambda_var.numpy() if lambda_var else [1.0]
    self.lambda_var = tf.constant(lambda_value, name="lambda")

    self.model = CustomValueNetwork(
      self.environment.observation_spec(),
      fc_layer_params=self.model_size,
      classifier=self.classifier,
      lambda_var=self.lambda_var
    )