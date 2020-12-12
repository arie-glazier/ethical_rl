from dataclasses import dataclass, asdict
from typing import List
import random, time

from ethical_rl.constants import *

@dataclass
class EpisodeResult:
  total_reward: int
  total_constraint_violations: int
  actions: List[int]

@dataclass
class LabeledData:
  label: str
  left_trajectory: List[dict]
  right_trajectory: List[dict]

class LabelerBASE:
  def __init__(self, **kwargs):
    self.environment = kwargs[ENVIRONMENT]

  def make_episode_pairs(self, episode_history):
    # Randomly shuffle episodes
    random.shuffle(episode_history)
    # Split list in half (TODO: should check to make sure we have an even number of episdoes)
    episodes_left = episode_history[:len(episode_history) // 2]
    episodes_right = episode_history[len(episode_history) // 2:]
    # Create pairs.  Result is an iterator where each item is (episode_result_a, episode_result_b)
    episode_pairs = zip(episodes_left, episodes_right)

    return episode_pairs

  def get_trajectory(self, episode, display=False):
    trajectory = []
    state = self.environment.reset()
    if display: self.environment.render()
    for action in episode.actions:
      if display: time.sleep(.1)
      trajectory.append((state,action))
      state, reward, done, info = self.environment.step(action)
      if display: self.environment.render()
    return trajectory