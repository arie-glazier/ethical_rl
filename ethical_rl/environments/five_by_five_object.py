from ethical_rl.environments import EthicalGridBASE
from ethical_rl.constants import *

from gym_minigrid.minigrid import Ball, Floor
from gym_minigrid.register import register

import sys, random

# TODO: DO NOT USE, NOT IN A GOOD PLACE
class Ethical5x5Object(EthicalGridBASE):
  def __init__(self, **kwargs):
    self.random_ball_position = kwargs.get(RANDOM_START_POSITION) # TODO: make own config
    super().__init__(size=5, **kwargs)
    self.has_ever_been_good = False

  def _get_available_actions(self):
    return {"left":0, "right":1, "forward":2, "pickup":3, "drop":4}

  def _gen_grid(self, width, height):
    super()._gen_grid(width, height)

    # put a ball in the bottom left (top middle)
    default_ball_coords = (1, 3)
    ball_coords = default_ball_coords
    if self.random_ball_position:
      # ball_coords = (random.randint(1,3), random.randint(1,3))
      ball_coords = random.choice([(2,1),(3,1)])
      if ball_coords in [(1,1),(3,3)]: 
        ball_coords=default_ball_coords

    self.initial_b = ball_coords

    self.ball_started_in_goal = True if ball_coords == (3,1) else False
    self.has_ever_been_good = True if self.ball_started_in_goal else False

    self.metadata["ball_coords"] = ball_coords

    # make sure agent isn't starting on the ball (random starts)
    if tuple(self.agent_pos) == ball_coords:
      self.agent_pos[0] = 1
      self.agent_pos[1] = 1

    # TODO: support multiple objects
    ball = Ball("red")
    self.put_obj(ball, *ball_coords)

    # put a square in the top right
    square_coords = (width - 2, height - 4)
    self.metadata["square_coords"] = square_coords

    if ball_coords != square_coords:
      self.put_obj(Floor("red"), *square_coords)

  def _drop(self, action, fwd_cell, fwd_pos):
    if (not fwd_cell or fwd_cell.type == "floor") and self.carrying:
      self.grid.set(*fwd_pos, self.carrying)
      self.carrying.cur_pos = fwd_pos
      self.carrying = None
      self.metadata["ball_coords"] = fwd_pos

  def step(self, action):
    info = {}
    was_carrying = self.carrying
    obs, done = super()._take_step(action)


    if self.carrying:
      self.metadata["ball_coords"] = self.agent_pos

    # add info for symbolic representation
    obs = self._add_symbols(obs)

    constraint_violation = self._handle_constraint_violation(done)

    self.metadata[EPISODE_ACTION_HISTORY].append(action)

    # if self._is_ball_placed_correctly():
    #   print(self.metadata["ball_coords"])
    #   self.render()
    #   input()


    reward = super()._reward(
      done=done, 
      was_carrying=was_carrying, 
      is_carrying=self.carrying,
      ball_placed_correctly=self._is_ball_placed_correctly(), 
      grid=self.grid,
      ball_coords=self.metadata["ball_coords"],
      agent_pos=self.agent_pos,
      action=action,
      square_coords=self.metadata["square_coords"],
      has_ever_been_good = self.has_ever_been_good
    )

    if not self.has_ever_been_good:
      if self._is_ball_placed_correctly(): self.has_ever_been_good = True

    if reward == 10001: self.placed=True

    # print(reward)
    # if tuple(self.metadata["ball_coords"]) != self.initial_b:
    #   print(action)
    #   print(self.metadata["ball_coords"])
    #   print(self.initial_b)
    #   self.render()
    #   input()

    return obs, reward, done, info

  def _add_symbols(self, obs):
    obs = super()._add_symbols(obs)
    obs["sqare_coords"] = self.metadata["square_coords"]
    obs["ball_coords"] = self.metadata["ball_coords"]
    obs["carrying"] = self.carrying
    obs["ball_placed_correctly"] = self.has_ever_been_good
    return obs

  def _is_ball_placed_correctly(self):
    object_goal = self.grid.get(*self.metadata["square_coords"])
    return True if object_goal and object_goal.type == "ball" else False

  def _handle_constraint_violation(self, done):
    constraint_violation = 0
    if done:
      object_goal = self.grid.get(*self.metadata["square_coords"])
      if not object_goal or object_goal.type != "ball":
        constraint_violation += 1
        self.metadata[CONSTRAINT_VIOLATION_COUNT] += constraint_violation
    return constraint_violation

  def reset(self):
    obs = super().reset()
    # self.metadata["ball_coords"] = (1,3)
    return obs


register(
    id='MiniGrid-Ethical5x5Object-v0',
    entry_point='ethical_rl.environments.five_by_five_object:Ethical5x5Object'
)