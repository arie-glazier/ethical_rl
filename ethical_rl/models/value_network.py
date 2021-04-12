from tf_agents.networks.value_network import ValueNetwork
from ethical_rl.constants import *

# this is the value network for PPO
class Model:
  def __init__(self, **kwargs):
    self.environment = kwargs[ENVIRONMENT]

    self.model = ValueNetwork(
      self.environment.observation_spec(),
      fc_layer_params=kwargs[FULLY_CONNECTED_MODEL_SIZE]
    )