import pytest

def test_lr_model(algorithm, test_sentence, results):
  algorithm.train()
  prediction = algorithm.test([test_sentence])
  assert prediction[0] == 1
  prediction = algorithm.test([test_sentence.replace("positive","negative")])
  assert prediction[0] == -1