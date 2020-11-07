import matplotlib.pyplot as plt
from services.constants import *
import pickle

class Reporting:
  def __init__(self, data_object,**kwargs):
    self.data_object = data_object
    self.results_destination = kwargs[RESULTS_DESTINATION]
    self.model = kwargs[MODEL_MODULE]
    self.environment_name = kwargs[ENVIRONMENT_NAME]
    self.replay_name = kwargs[REPLAY_BUFFER_MODULE]
    self.algorithm_name = kwargs[ALGORITHM_MODULE]
    self.test_name = kwargs[TEST_NAME]
    self.number_of_episodes = kwargs[NUMBER_OF_EPISODES]
    self.env = kwargs[ENVIRONMENT]


  def __create_reward_graph(self):
    plt.plot(self.data_object['return'])
    plt.xlabel("episodes")
    plt.ylabel("total return")
    plt.savefig(f"{self.results_destination}reward_graph")
    
  def __create_constraint_graph(self):
    plt.plot(self.data_object['performance'])
    plt.xlabel("episodes")
    plt.ylabel("performance")
    plt.savefig(f"{self.results_destination}reward_graph")

  def __dump_data(self):
    title = f"{self.test_name}_{self.algorithm_name}_{self.model_name}_{self.replay_name}_{self.number_of_episodes}"
    pickle.dump(self.data_object, open(f"{self.results_destination}{title}.pickle", "wb"))