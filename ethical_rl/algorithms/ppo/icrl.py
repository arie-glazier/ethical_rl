from ethical_rl.algorithms import AlgorithmBASE
from ethical_rl.algorithms.ppo import Algorithm as PpoAlgorithm
# from ethical_rl.reward_predictor import RewardPredictor as Classifier
import tensorflow as tf
import pickle, sys, os, time
import numpy as np
import tensorflow_constrained_optimization as tfco
from ethical_rl.util import load_model

def one_hot_map(attribute_size, attribute_observation):
  array = np.zeros(attribute_size)
  array[attribute_observation] = 1
  return array

class Algorithm(AlgorithmBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.input_shape = 10
    self.kwargs = kwargs

    self.classifier = self._classifier_model()
    self.value_net = load_model("ethical_rl.models.value_network")(**kwargs, classifier=self.classifier).model
    self.ppo_algo = PpoAlgorithm(**{**kwargs,**{"model":self.value_net, "number_of_episodes":int(kwargs["policy_training_steps"])}})

    self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
    self.loss_function = tf.keras.losses.BinaryCrossentropy()
    self.lambda_var = tf.Variable(float(kwargs["initial_lambda_value"]), name="Lambda", constraint=lambda x: tf.clip_by_value(x, 0.0, 1000.0))

    self.classifier_training_steps = int(kwargs["classifier_training_steps"])
    self.rollout_length = int(kwargs["rollout_length"])

    # self.lambda_optimizer = tf.keras.optimizers.SGD(learning_rate=float(kwargs["lambda_learning_rate"]))
    self.lambda_optimizer = tf.keras.optimizers.Adam(learning_rate=float(kwargs["lambda_learning_rate"]))

    self.alpha = tf.constant(0.01) # not same as alpha in value_network. differnce between multiplying and individual

    self.dataset = self.ppo_algo.replay_buffer.as_dataset(
      num_parallel_calls=1,
      sample_batch_size=1,
      num_steps=10
    ).prefetch(1)
    self.iterator = iter(self.dataset)

    self.expert_trajectories, self.expert_labels = self._load_expert_trajectories()


  def _freeze_model(self, model):
    cloned_classifier = tf.keras.models.clone_model(model)
    cloned_classifier.set_weights(model.get_weights())
    cloned_classifier.trainable = False
    cloned_classifier.compile()
    return cloned_classifier

  def _mock_constraint_data(self):
    possible_directions = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    constrained_spaces = [[0,0,1,1,0,0], [1,0,0,0,0,1]]
    augmented_data = []
    for location in constrained_spaces:
      location_vec = np.array(location)
      for direction in possible_directions:
        dir_vec = np.array(direction)
        augmented_data.append(np.concatenate((dir_vec, location_vec)))

    sample_data = np.unique(augmented_data, axis=0)
    sample_labels = np.ones((sample_data.shape[0],)) * -1

    return sample_data, sample_labels

  def _learn_lambda(self, training_experience):
    training_observations = []
    for x in training_experience:
      training_observations.extend(x.observation.numpy())

    for trajectory in training_observations:
      training_predictions = self.classifier.predict(trajectory)
      cost_function = tf.constant(1.0) - training_predictions
      trajectory_prediction = tf.math.reduce_prod(cost_function)

      loss = lambda: (-1.0 * (self.lambda_var ** 2) * (cost_function - self.alpha)) / 2.0 #d(loss)/d(lambda_var) = - lambda_var (trajectory_prediction - alpha)
      self.lambda_optimizer.minimize(loss, [self.lambda_var])
    return 

  def _train_policy(self):
    lambda_constant = tf.constant(self.lambda_var.numpy())
    self.ppo_algo.value_net.lambda_var = lambda_constant
    frozen_classifier = self._freeze_model(self.classifier)
    self.ppo_algo.value_net.classifier = frozen_classifier
    training_experience = self.ppo_algo.train()
    return training_experience

  def train(self):
    for episode in range(self.number_of_episodes):
      print(f"episode: {episode} / lambda: {self.lambda_var.numpy()}")
      training_experience = self._train_policy()
      self._learn_lambda(training_experience)

      skipped_counter = 0
      for _ in range(self.classifier_training_steps):
        # sample_data, sample_labels = self._sample_trajectories()
        # just get things working with this
        sample_data, sample_labels = self._mock_constraint_data()

        if sample_data.shape[0] == 0:
          print("skipping classifier training - no samples violate constraints")
          skipped_counter += 1
          if skipped_counter >= 100:
            break
          else:
            continue

        expert_trajectories = self.expert_trajectories #[np.random.randint(self.expert_trajectories.shape[0],size=sample_data.shape[0]*2),:]
        train_data = np.concatenate((expert_trajectories, sample_data))
        expert_labels = self.expert_labels #[np.random.randint(self.expert_labels.shape[0],size=sample_data.shape[0]*2)]
        train_labels = np.concatenate((expert_labels, sample_labels))

        self.classifier.fit(train_data, train_labels, epochs=1000, verbose=0)

        # good_state = tf.convert_to_tensor(np.array([[1,0,0,0,1,0,0,1,0,0]]))
        # bad_state = tf.convert_to_tensor(np.array([[1,0,0,0,0,0,1,1,0,0]]))
        # test_states = train_data[[1,-1]]
        # print(test_states)
        # print(self.classifier.predict(test_states))
    input("display")
    self._display_results()

  def _display_results(self):
    for test_no in range(5):
      time_step = self.ppo_algo.env.reset()
      self.ppo_algo.env._env.envs[0].gym.render()
      r = 0
      for step in range(7):
        action = self.ppo_algo.tf_agent.policy.action(time_step)
        time_step = self.ppo_algo.env.step(action.action)
        self.ppo_algo.env._env.envs[0].gym.render()
        r += time_step.reward
        time.sleep(0.1)
      print(f"reward: {r}")

  def _get_importance_weights(self, states):
    predictions = self.classifier.predict(states)[0]
    weights = predictions / (1-predictions)
    return weights

  def _classifier_model(self):
    input_layer = tf.keras.layers.Dense(5, activation="relu", input_shape=(self.input_shape,), name="classifier_input")
    hidden_layer = tf.keras.layers.Dense(3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), name="classifier_hidden")
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier_output")
    model = tf.keras.models.Sequential([
      input_layer,
      hidden_layer,
      output_layer
    ])

    model.compile(loss=self.loss_function, optimizer=self.optimizer)
    return model

  def _load_expert_trajectories(self):
    with open(os.path.join(os.getcwd(), "data","labels","labeled_data_synthetic"), "rb") as f: self.data = pickle.load(f)

    # start with goal states
    expert_trajectories = [
      np.concatenate((one_hot_map(4,0), one_hot_map(3,2), one_hot_map(3,2))),
      np.concatenate((one_hot_map(4,1), one_hot_map(3,2), one_hot_map(3,2))),
      np.concatenate((one_hot_map(4,2), one_hot_map(3,2), one_hot_map(3,2))),
      np.concatenate((one_hot_map(4,3), one_hot_map(3,2), one_hot_map(3,2)))
    ]
    for episode in self.data:
      for step in episode["left_trajectory"]:
        direction = one_hot_map(4,step[0]["direction"])
        x = one_hot_map(3,step[0]["agent_position"][0]-1)
        y = one_hot_map(3,step[0]["agent_position"][1]-1)
        state = np.concatenate((direction, x, y))
        if step[0]["constraint_violation_count"] > 0: 
          break
        expert_trajectories.append(state)
      for step in episode["right_trajectory"]:
        if step[0]["constraint_violation_count"] > 0: break
        direction = one_hot_map(4,step[0]["direction"])
        x = one_hot_map(3,step[0]["agent_position"][0]-1)
        y = one_hot_map(3,step[0]["agent_position"][1]-1)
        state = np.concatenate((direction, x, y))
        expert_trajectories.append(state)

    # self.expert_trajectories = np.array(expert_trajectories)
    # self.expert_trajectories = np.unique(self.expert_trajectories, axis=0)
    augmented_data = []
    possible_directions = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    for sample in expert_trajectories:
      augmented_data.append(sample)
      for direction in possible_directions:
        dir_vec = np.array(direction)
        augmented_data.append(np.concatenate((dir_vec, sample[4:])))

    expert_trajectories = np.unique(augmented_data, axis=0)
    expert_labels = np.ones((expert_trajectories.shape[0],))

    return expert_trajectories, expert_labels
  
  def _sample_trajectories(self):
    self.ppo_algo.replay_buffer.clear()
    for _ in range(self.rollout_length):
      self.ppo_algo._collect_step(self.ppo_algo.env, self.ppo_algo.tf_agent.collect_policy, self.ppo_algo.replay_buffer)
    experience, unused_info = next(self.iterator)

    sample_states = experience.observation
    importance_sampling_weights = self._get_importance_weights(sample_states)
    sample_data = np.array([x for x in sample_states.numpy()[0] if x.tolist() not in self.expert_trajectories.tolist()])
    sample_data = [x for x in sample_states.numpy()[0] if x.tolist() not in self.expert_trajectories.tolist()]
    sample_data = np.unique(sample_data, axis=0).tolist()
    possible_directions = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    augmented_data = []
    for sample in sample_data:
      augmented_data.append(sample)
      for direction in possible_directions:
        dir_vec = np.array(direction)
        augmented_data.append(np.concatenate((dir_vec, sample[4:])))

    sample_data = np.unique(augmented_data, axis=0)
    sample_labels = np.ones((sample_data.shape[0],)) * -1

    return sample_data, sample_labels