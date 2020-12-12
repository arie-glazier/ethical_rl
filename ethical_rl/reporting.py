import matplotlib.pyplot as plt
from ethical_rl.constants import *
import pickle

class Reporting:
  def __init__(self, data_object):
    self.data_object = data_object
    self.config = data_object[CONFIG]
    self.results_destination = self.config[RESULTS_DESTINATION]
    self.model_name = self.config[MODEL_MODULE]
    self.replay_name = self.config[REPLAY_BUFFER_MODULE]
    self.algorithm_name = self.config[ALGORITHM_MODULE]
    self.test_name = self.config[TEST_NAME]
    self.number_of_episodes = self.config[NUMBER_OF_EPISODES]


  def create_graph(self,data_object_key, x_label, y_label,title = None, **kwargs):
    plt.plot(self.data_object[data_object_key],**kwargs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{self.results_destination}{data_object_key}_graph")
    plt.clf()

  def dump_data(self):
    title = f"{self.test_name}_{self.algorithm_name}_{self.model_name}_{self.replay_name}_{self.number_of_episodes}"
    pickle.dump(self.data_object, open(f"{self.results_destination}{title}.pickle", "wb"))