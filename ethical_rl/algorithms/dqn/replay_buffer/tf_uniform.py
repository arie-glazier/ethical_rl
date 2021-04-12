from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from ethical_rl.constants import *

# TODO: replay buffers should be moved out of dqn folder
class ReplayBuffer:
  def __init__(self, **kwargs):
    self.batch_size = 1
    self.tf_agent = kwargs["tf_agent"]

    self.replay_buffer = TFUniformReplayBuffer(
      self.tf_agent.collect_data_spec, 
      batch_size=self.batch_size, 
      max_length=kwargs[MAX_REPLAY_BUFFER_LENGTH]
    )