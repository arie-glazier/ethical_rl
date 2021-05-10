import gym, sys
from gym.envs.registration import register
import numpy as np

class Environment(gym.Env):
  def __init__(self, **kwargs):
    self.n_features = kwargs["n_features"]
    self.n_actions = kwargs["n_actions"]

    # a state is a user, represented by embeddings of
    # all topics ever clicked on (can later expand to include 
    # other things like open history, click history, etc.)
    self.observation_space = gym.spaces.Box(
      low=0, 
      high=1,
      shape=(1,self.n_features)
      )

    # an action is an article, similar rep as state
    self.action_space = gym.spaces.Box(
      low=0, 
      high=1,
      shape=(1,self.n_actions)
      )

    # this is (user_doc)
    self.agent_state = None

    self.state_iterator = None

    self.history = kwargs["history"]

    self.step_counter = 0

    self.max_sim = 0
    self.min_sim = 0

  def _set_state_iterator(self):
    # get random person_guid
    sample = self.history.sample(1)
    person_history = self.history[(self.history.person_guid==sample.iloc[0].person_guid)]
    person_history = person_history.sort_values(by=["event_rank"])
    for i,row in person_history.iloc[1:].iterrows():
        yield row

  def _get_next_state(self):
    try:
      return next(self.state_iterator)
    except:
      return self.agent_state # this is bad

  def _calc_sim(self, a,b,threshold=0.1):
    min_len = min(a.shape,b.shape)#[0]
    if type(min_len) == tuple: min_len=min_len[0]
    a = a[:min_len]
    b = b[:min_len]
    value = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if value > self.max_sim:
      self.max_sim = value
    if value < self.min_sim:
      self.min_sim = value
    return value 

  def _reward(self, state, action):
    return int(self._calc_sim(state.action_context, action.iloc[0]) * 10) - 1

  def reset(self):
    # needs to return a value within self.observation_space
    # can be an integer if descrete, can be numpy.array if box
    self.state_iterator = self._set_state_iterator()
    self.agent_state = self._get_next_state()
    self.step_counter = 0
    self.max_sim = 0
    self.min_sim = 0
    return self.agent_state.context

  def step(self, action):
    # action needs to be a value within self.action_space
    # return is a 4-tuple: (state, reward, done, info)
    self.step_counter += 1

    info = {}

    reward = self._reward(self.agent_state, action) #1 / (np.linalg.norm(self.agent_state - action) + 0.1)
    
    if reward >= 8:
      done = self.agent_state.last_click
      next_state = self._get_next_state()
      if next_state is not None:
        self.agent_state = next_state
    else:
      done = False
      next_state = self.agent_state

    if done: 
      reward += 1000 / self.step_counter

    return (next_state.context if next_state is not None else self.agent_state.context, reward, done, info)

  # additional optional methods: render, close, seed

register(
    id='RecBasic-v0',
    entry_point='ethical_rl.environments.recommendation.basic:Environment'
)