import pytest

@pytest.fixture(scope="session")
def train_sentences():
  return [
    "This is a positive sentence!",
    "This is a negative sentence",
    "Candy is a positive thing!",
    "Vegetables are a negative thing!"
  ]

@pytest.fixture(scope="session")
def test_sentence():
  return "I really hope this is a positive!"

@pytest.fixture(scope="session")
def test_labels():
  return [
    1,
    -1,
    1,
    -1
  ]