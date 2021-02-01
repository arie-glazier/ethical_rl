import numpy as np
import sys
from ethical_rl.constants import *
from scipy.spatial import distance
import tensorflow as tf

class Policy:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]
    self.n_outputs = self.env.action_space.n

  def get_action(self, model, state, epsilon):
    # TODO: make possible actions passed in
    # available_actions = [0,1,2,3,4]
    # directions = state[0:4]
    # x = state[4:7]
    # y = state[7:10]
    # bx = state[10:13]
    # by = state[13:]
    # is_carrying = np.array_equal(x,bx) and np.array_equal(y,by)
    # can_pickup = distance.cdist([(np.argmax(x),np.argmax(y))], [(np.argmax(bx), np.argmax(by))], "cityblock")[0][0] == 1
    # ball_in_goal = bx[0] == 1 and by[2] == 1
    # if is_carrying: available_actions = [0,1,2,4]
    # elif can_pickup: available_actions = [0,1,2,3]
    # if ball_in_goal: available_actions = [0,1,2]

    # if np.random.rand() < epsilon:
    #   return np.random.choice(available_actions) 
    # else:
    #   masked_prediction = tf.reduce_sum(model.predict(state[np.newaxis])[0] * tf.one_hot(available_actions, self.n_outputs), 0)
    #   masked_prediction = tf.where(tf.equal(masked_prediction,0), tf.multiply(tf.ones_like(masked_prediction), tf.constant(-100.0)), masked_prediction)
    #   return np.argmax(masked_prediction)


    # return np.random.choice(available_actions) if np.random.rand() < epsilon else np.argmax(model.predict(state[np.newaxis])[0])
    return np.random.randint(self.n_outputs) if np.random.rand() < epsilon else np.argmax(model.predict(state[np.newaxis])[0])