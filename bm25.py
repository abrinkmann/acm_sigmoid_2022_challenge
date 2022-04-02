"""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.75, k1=1.6, min_df=1, max_df=1.0, tokenizer=None):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, min_df=min_df, max_df=max_df, tokenizer=tokenizer)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        self.X_transformed = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.len_X = self.X_transformed.sum(1).A1
        self.avdl = self.X_transformed.sum(1).mean()

    def transform(self, q):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl, X_transformed, len_X = self.b, self.k1, self.avdl, self.X_transformed, self.len_X

        # apply CountVectorizer
        #X_transformed = super(TfidfVectorizer, self.vectorizer).transform(X)
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X_transformed = X_transformed.tocsc()[:, q.indices]
        denom = X_transformed + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X_transformed.multiply(np.broadcast_to(idf, X_transformed.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

