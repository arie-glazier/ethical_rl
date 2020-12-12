from ethical_rl.common.schedules import ScheduleBase

class Schedule(ScheduleBase):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v