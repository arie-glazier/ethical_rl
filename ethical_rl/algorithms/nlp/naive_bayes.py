import math
from ethical_rl.algorithms.nlp import NlpUtils
import pandas as pd

class Algorithm:
  '''
  Data: D = {(x0,y0),...,(xn,yn)}

  Idea: estimate p(y|x) and compare p(y=1|x) > p(y=-1|x)

  Bayes Rule: p(y|x) = p(x|y) * p(y) / p(x)
  + Prior: p(y) => p(y=1) = count(y=1) / count(y=1) + count(y=-1)
  + Evidence: p(x) => marginalize over vocab
  + Likelihood: p(x|y)
    + Problem - O(k2^d) without conditional independence assumption
    + Tricky to estimate, which is at the heart of what we're doing here

  Because of conditional independence assumption we get:
  + p(y=1 | x) = p(y=1) * product_j(p(xij|y=1)) / p(x)

  When a word comes in that is missing from the vocab prob = 0
  +  Need to add smoothing (common usage is epsilon=1 in numerator and 2e in denom)

  Multinomial Case:
  + p(xij|yi) = Tij / sum(Tik) where Tcj is the number of term j in documents of class c
  + p(yi|xi) sums over *tokens* rather than terms as in the binomial case
  + Intuition:
    * Bernoulli model - p(x_j|y) is the fraction of *documents* in class y containing term j
    * Multinomial model - p(x_y|y) is the fraction of *tokens* in class y that contain term j 
  + smoothing denom => |V| * epsilon
  '''
  def __init__(self, **kwargs):
    self.documents = kwargs["documents"]
    self.labels = kwargs["labels"]

    self.tokenized_docs = [ NlpUtils.tokenize(d) for d in self.documents ]
    self.features = [NlpUtils.binary_featurize(tokens) for tokens in self.tokenized_docs ]

    self.word_counts = NlpUtils.count_word_document_frequency(self.features)
    self.vocabulary = NlpUtils.create_vocabulary(self.word_counts)
    self.pruned_features = [ NlpUtils.prune_features(self.vocabulary, f) for f in self.features ]

    self.train_df = pd.DataFrame({"features":self.pruned_features, "label":self.labels})

    self.likelihoods = self._get_likelihoods()
    self.priors = {label:self.compute_prior(label) for label in self.labels}

  def _get_likelihoods(self, epsilon=1):
    return {label:self.p_x_given_y(label, epsilon) for label in self.labels}

  def p_x_given_y(self, label, epsilon=1):
    labeled_df = self.train_df[self.train_df.label == label]
    labeled_word_counts = NlpUtils.count_word_document_frequency(labeled_df.features)
    total_documents = labeled_df.label.count()

    result = {}
    for word in self.vocabulary:
        count = labeled_word_counts.get(word) or 0
        result[word] = (epsilon + count) / (2 * epsilon + total_documents)
    return result

  def compute_prior(self, label):
    return self.train_df[self.train_df.label==label].label.count() / self.train_df.label.count()

  def compute_log_numerator(self, features, label):
    likelihood = 0
    for word in self.vocabulary:
      probability = self.likelihoods[label][word] if word in features.keys() else 1 - self.likelihoods[label][word]
      l = self.likelihoods[label]
      likelihood += math.log(probability)

    return math.log(self.priors[label]) + likelihood

  def classify(self, document, label=True):
    document_tokens = NlpUtils.tokenize(document)
    features = NlpUtils.binary_featurize(document_tokens)
    pruned_features = NlpUtils.prune_features(self.vocabulary, features)
    # print(pruned_features)
    numerators = {label:self.compute_log_numerator(pruned_features, label) for label in self.labels }
    # there's an additional log-sum-exp trick to avoid underflow when computing p(x)
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    maxv = max(numerators.values())
    log_p_x = maxv + math.log(sum([math.exp(v-maxv) for v in numerators.values()]))
    v = numerators[label] - log_p_x
    return math.exp(v)
      