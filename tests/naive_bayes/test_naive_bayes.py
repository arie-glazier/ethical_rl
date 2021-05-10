import pytest

def test_vocabulary(algorithm, results):
  assert algorithm.vocabulary == results["vocabulary"]

def test_p_x_given_y(algorithm, results):
  positive_scores = algorithm.p_x_given_y(1) 
  assert positive_scores == results["pos_p_x_given_y"]
  negative_scores = algorithm.p_x_given_y(-1)
  assert negative_scores == results["neg_p_x_given_y"]

  print(positive_scores)

def test_prior(algorithm, results):
  positive_prior = algorithm.compute_prior(1)
  assert positive_prior == results["positive_prior"]

  negative_prior = algorithm.compute_prior(-1)
  assert negative_prior == results["negative_prior"]

def test_classify(algorithm, test_sentence, results):
  algorithm.likelihoods = algorithm._get_likelihoods(epsilon=1)
  positive_probability = algorithm.classify(test_sentence, 1)
  assert round(positive_probability,2) == 0.93

