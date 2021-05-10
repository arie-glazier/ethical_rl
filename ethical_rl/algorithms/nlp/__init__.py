'''
True/False
Given equations, compute output
Given code, explain what is being computed
Given model, suggest an modification to address a problem
Given a problem, design a model

compute classification output of logistic regression, naive bayes, neural net
compute parameter estimate / step of gradient descent given equations
understand dimensions of weight matrices in neural nets
viterbi / forward-backward for HMMs
design choices in language models
pros / cons of language models, RNNs, LSTMs
'''
from collections import Counter
import re
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB,MultinomialNB

class NlpUtils:

  @staticmethod
  def featurize(tokens):
    return { token:1 for token in tokens}

  @staticmethod
  def tokenize(document, strip_punct=True, ignore_case=True):
    if ignore_case: document = document.lower()
    tokens = document.split()
    if strip_punct: 
      # tokens = [re.sub("^\W*|\W*$","",x.replace("'","")) for x in tokens]
      tokens = [re.sub("[^\w\s]","",x) for x in tokens]
    return tokens

  @staticmethod
  def binary_featurize(tokens):
    return { token:1 for token in tokens}

  @staticmethod
  def count_word_document_frequency(dict_list):
    return Counter([key for keys in [ d.keys() for d in dict_list] for key in keys if key])

  @staticmethod
  def create_vocabulary(word_counts, min_count=5, max_count=100):
    result = {}
    identifier = 0
    for word in sorted(word_counts):
        count = word_counts.get(word)
        if count >= min_count and count <= max_count:
            result[word] = identifier
            identifier += 1
    return result

  @staticmethod
  def prune_features(vocabulary, raw_feature_dict):
    return {k:v for k,v in raw_feature_dict.items() if k in vocabulary}

  @staticmethod
  def features2array(features, vocabulary):
    result = []
    for word in vocabulary:
        result.append(1 if features.get(word) else 0)
    return np.array(result)

  @staticmethod
  def features2sparse_array(features, vocabulary):
    return csr_matrix(features2array(features, vocabulary), shape=(1, len(vocabulary)))

  @staticmethod
  def make_count_vectorizer(training_text, vectorizer_type=CountVectorizer):
    vectorizer = vectorizer_type(max_df=100,min_df=2,binary=True)
    vectorizer.fit(training_text)
    return vectorizer