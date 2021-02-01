import pickle, sys, argparse, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from ethical_rl.reward_predictor import RewardPredictor

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--pickle_file", default="./data/labels/labeled_data_synthetic")
PARSER.add_argument("--min_mu", default=-5.0, type=float)
PARSER.add_argument("--max_mu", default=-1.0, type=float)
PARSER.add_argument("--model_save_folder", default="./saved_models")
PARSER.add_argument("--model_save_name")
PARSER.add_argument("--number_of_training_observations", type=int)

if __name__ == "__main__":
  print("\n *** Training Reward Predictor *** \n")
  args = PARSER.parse_args()
  min_mu = float(args.min_mu)
  max_mu = float(args.max_mu)
  model_save_path = os.path.join(args.model_save_folder, args.model_save_name) if args.model_save_name  else None
  number_of_training_observations = args.number_of_training_observations if args.number_of_training_observations else 0

  print("DATA:")
  with open(args.pickle_file, "rb") as f: data = pickle.load(f)
  print(f"  + loaded data: {len(data)} observations")
  training_data_size = number_of_training_observations if number_of_training_observations >= 0 else len(data)
  training_data = np.random.choice(data, training_data_size, replace=False) # for ablation studies
  # training_data = data[4:]
  print(f"  + training set: {len(training_data)} observations \n")

  # training data needs to be in the form (sigma1, sigma2, mu)
  # where sigmas are trajectories and mu = -1 if label is left,
  # -2 if label is right, and -1.5 if label is same
  # sigma = [(state0,action0),...,(stateN,actionN)]
  print("MODEL:")
  reward_predictor = RewardPredictor(min_mu=min_mu, max_mu=max_mu)
  print("  + processing trajectory")
  for idx, item in enumerate(training_data): 
    mu = reward_predictor.mu(item["label"])

    if mu:
      reward_predictor.process_trajectory(item["left_trajectory"], 0, item["label"]) #TODO: mu_idx not good
      reward_predictor.process_trajectory(item["right_trajectory"], 1, item["label"])

  print("  -> done")

  # print(set(reward_predictor.training_labels))
  # sys.exit()
  # for x,y in zip(reward_predictor.training_data, reward_predictor.training_labels):
  #   print(f"{y} : {x}")
  #   if y == -1.5: sys.exit()
  # sys.exit()

  print("  + training model")
  model = reward_predictor.initialize_model()
  if reward_predictor.training_data: model = reward_predictor.train_model(model)
  print("  -> done")
  if model_save_path: 
    print(f"  + saving model to: {model_save_path}")
    reward_predictor.save_model(model, model_save_path)
    print("  -> done \n")

  # testing output
  # optimal path:
  reward = 0
  goal_value = 0
  direction, x_position, y_position, action = 0, 0, 0, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 0, 1, 0, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 0, 2, 0, 1
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 1, 2, 0, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 1, 2, 1, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action) + goal_value
  print(f"reward for optimal path is: {reward}")
  # constraint path:
  reward = 0
  direction, x_position, y_position, action = 0, 0, 0, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 0, 1, 0, 1
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 1, 1, 0, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 1, 1, 1, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 1, 1, 2, 0
  reward += reward_predictor.predict(model, direction, x_position, y_position, action)
  direction, x_position, y_position, action = 0, 1, 2, 2
  reward += reward_predictor.predict(model, direction, x_position, y_position, action) + goal_value
  print(f"reward for constraint path is: {reward}")