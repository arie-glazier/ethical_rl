import pytest
from ethical_rl.algorithms.nlp.naive_bayes import Algorithm

@pytest.fixture(scope="session")
def algorithm(train_sentences, test_labels):
  return Algorithm(
    documents=train_sentences,
    labels=test_labels
  )

@pytest.fixture(scope="session")
def results():
  return {
    "vocabulary":{'a': 0, 'is': 1, 'negative': 2, 'positive': 3, 'sentence': 4, 'thing': 5, 'this': 6},
    "pos_p_x_given_y":{'a': 1.0, 'is': 1.0, 'negative': 0.0, 'positive': 1.0, 'sentence': 0.5, 'thing': 0.5, 'this': 0.5},
    "neg_p_x_given_y":{'a': 1.0, 'is': 0.5, 'negative': 1.0, 'positive': 0.0, 'sentence': 0.5, 'thing': 0.5, 'this': 0.5},
    "positive_prior": 0.5,
    "negative_prior": 0.5,
  }