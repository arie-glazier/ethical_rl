import matplotlib.pyplot as plt
from services.constants import *
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


  def create_return_graph(self):
    plt.plot(self.data_object['returns'])
    plt.xlabel("episodes")
    plt.ylabel("total return")
    plt.savefig(f"{self.results_destination}return_graph")
    plt.clf()
    
  def create_constraint_violation_graph(self):
    plt.plot(self.data_object['constraint_violations'])
    plt.xlabel("episodes")
    plt.ylabel("constraint violations")
    plt.savefig(f"{self.results_destination}constraint_violation_graph")
    plt.clf()

  def create_performance_graph(self):
    plt.plot(self.data_object['performance'])
    plt.xlabel("episodes")
    plt.ylabel("performance")
    plt.savefig(f"{self.results_destination}performance_graph")
    plt.clf()

  def dump_data(self):
    title = f"{self.test_name}_{self.algorithm_name}_{self.model_name}_{self.replay_name}_{self.number_of_episodes}"
    pickle.dump(self.data_object, open(f"{self.results_destination}{title}.pickle", "wb"))