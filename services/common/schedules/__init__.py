import sys
# whole folder jacked from: https://github.com/openai/baselines/blob/master/baselines/common/schedules.py
# TODO: Go through with a fine toothed comb to make sure everything is good.
"""This file is used for specifying various schedules that evolve over
time throughout the execution of the algorithm, such as:
 - learning rate for the optimizer
 - exploration epsilon for the epsilon greedy exploration strategy
 - beta parameter for beta parameter in prioritized replay
Each schedule has a function `value(t)` which returns the current value
of the parameter given the timestep t of the optimization procedure.
"""

class ScheduleBase(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()