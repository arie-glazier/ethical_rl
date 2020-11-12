import pickle, sys, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from services.reward_predictor import RewardPredictor

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--pickle_file", default="labeled_data")
PARSER.add_argument("--min_mu", default=-2.0)
PARSER.add_argument("--max_mu", default=-1.0)

if __name__ == "__main__":
  args = PARSER.parse_args()
  min_mu = float(args.min_mu)
  max_mu = float(args.max_mu)

  with open(args.pickle_file, "rb") as f: data = pickle.load(f)

  # training data needs to be in the form (sigma1, sigma2, mu)
  # where sigmas are trajectories and mu = 1 if label is left,
  # 2 if label is right, and 1.5 if lable is same
  # sigma = [(state0,action0),...,(stateN,actionN)]
  reward_predictor = RewardPredictor(min_mu=min_mu, max_mu=max_mu)
  for idx, item in enumerate(data[4:]): # this can be changed, bad data from demonstrating human training
    mu = reward_predictor.mu(item["label"])

    if mu == None:
      print("we shouldnt be here")
      print(idx)
      print(item)
      sys.exit()

    reward_predictor.process_trajectory(item["left_trajectory"], 0, item["label"]) #TODO: mu_idx not good
    reward_predictor.process_trajectory(item["right_trajectory"], 1, item["label"])

  model = reward_predictor.train_model()

  direction, x_position, y_position, action = 0, 1, 1, 2
  reward_predictor.predict(model, direction, x_position, y_position, action)