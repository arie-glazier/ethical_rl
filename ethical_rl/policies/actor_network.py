from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from ethical_rl.constants import *

# This is the actor network for PPO
class Policy:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]
    self.model = ActorDistributionNetwork(
      self.env.observation_spec(),
      self.env.action_spec(),
      fc_layer_params=kwargs[FULLY_CONNECTED_MODEL_SIZE]
    )