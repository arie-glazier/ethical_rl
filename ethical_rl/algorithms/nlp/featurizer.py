import numpy as np

class Featurizer:
  def __init__(self, **kwargs):
    self.documents = kwargs["documents"]
    self.representation = kwargs["representation"]

class Vectorizer(Featurizer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.max_df = kwargs.get("max_df") or 100
    self.min_df = kwargs.get("min_df") or 2
    self.binary = kwargs.get("binary") or True
    self.n_gram_range = kwargs.get("n_gram_range") or (1,3)
    self.stop_words = kwargs.get("stop_words") or "english"

    self.vectorizer = self.representation(
      max_df = self.max_df,
      min_df = self.min_df,
      binary = self.binary,
      ngram_range  = self.n_gram_range,
      stop_words = self.stop_words
    )
    self.vectorizer.fit(self.documents)

  def featurize(self, documents, **kwargs):
    return self.vectorizer.transform(documents).toarray()

class Berter(Featurizer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.vectorizer = self.representation()
    self.combination_type = kwargs.get("combination_type") or "concat"
    self.max_bert_length = None

  def do_pad(self, x, max_doc_length):
    new_x = np.array(x)
    new_x.resize(max_doc_length)
    return new_x

  def concat(self, x):
    return np.concatenate(x, axis=None)

  def avg(self, x):
    return np.mean(x, axis=0)

  def featurize(self, documents, max_bert_length=None):
    raw_bert = documents.map(lambda x: self.vectorizer([x]))
    extracted_bert = raw_bert.map(lambda x: tuple(y for y in x[0][1]))

    combination_func = getattr(self, self.combination_type)
    combined_bert = extracted_bert.map(combination_func)

    if max_bert_length or not self.max_bert_length: 
      self.max_bert_length = max_bert_length or max(combined_bert.map(len))

    # add padding
    bert = combined_bert.map(lambda x: self.do_pad(x,self.max_bert_length))
    
    return np.vstack(bert)