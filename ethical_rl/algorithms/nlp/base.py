from sklearn.linear_model import LogisticRegression
from ethical_rl.algorithms.nlp import NlpUtils
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class AlgorithmBASE:
  def __init__(self, **kwargs):
    self.documents = kwargs["documents"]
    self.labels = np.array(kwargs["labels"])

    if kwargs.get("featurizer"):
      self.featurizer = kwargs["featurizer"]
      self.vectorizer = self.featurizer.vectorizer
      self.document_features = self.featurizer.featurize(self.documents)
    else:
      self.vectorizer_type = kwargs.get("vectorizer_type") or CountVectorizer
      self.vectorizer = NlpUtils.make_count_vectorizer(self.documents, self.vectorizer_type)
      self.document_features = self._featurize(self.documents)

    self.other_features = kwargs.get("other_features")
    if self.other_features:
      self.all_features = np.hstack((self.document_features, self.user_indicators))
    else:
      self.all_features = self.document_features

