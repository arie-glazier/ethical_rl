from sklearn.neural_network import MLPClassifier
from ethical_rl.algorithms.nlp import NlpUtils
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ethical_rl.algorithms.nlp.base import AlgorithmBASE

class Algorithm(AlgorithmBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.hidden_sizes = kwargs.get("hidden_sizes") or (100,)
    self.model = MLPClassifier(hidden_layer_sizes=self.hidden_sizes)

  def train(self, features, labels):
    self.model.fit(features, labels)

  def classify(self, test_document):
    featurized_doc = self.featurizer.featurize([test_document])
    return self.model.predict_proba(featurized_doc)
