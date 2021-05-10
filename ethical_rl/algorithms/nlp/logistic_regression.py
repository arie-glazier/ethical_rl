from sklearn.linear_model import LogisticRegression
from ethical_rl.algorithms.nlp import NlpUtils
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ethical_rl.algorithms.nlp.base import AlgorithmBASE

class Algorithm(AlgorithmBASE):
  '''
  Naive Bayes estimates p(y|x) by inverting the conditional p(x|y)
  Logistic Regression estimates p(y|x) directly
  '''
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.solver = kwargs.get("solver") or "liblinear"
    self.model = LogisticRegression(solver=self.solver)

  def _featurize(self, documents):
    return self.vectorizer.transform(documents).toarray()

  def train(self, features, labels):
    self.model.fit(features, labels)

  def classify(self, test_document):
    featurized_doc = self._featurize([test_document])
    return self.model.predict_proba(featurized_doc)
