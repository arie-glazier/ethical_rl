import pytest
from ethical_rl.algorithms.nlp.logistic_regression import Algorithm

@pytest.fixture(scope="session")
def algorithm(train_sentences, test_labels):
  return Algorithm(
    documents=train_sentences,
    labels=test_labels
  )

@pytest.fixture(scope="session")
def results():
  return {
  }