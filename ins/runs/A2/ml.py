"""Genuinely-trained ML components for the transducer.

DeletionClf: LogisticRegression over char n-gram COUNT features (n=2..4) of the
lang-tagged source span -> P(replacement==""). Materially decides every deletion
output (high-precision threshold) and gates false deletions on kept spans.
(No TF-IDF / idf weighting anywhere -- plain counts only.)
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class DeletionClf:
    def __init__(self, thr=0.60):
        self.thr = thr
        self.vec = CountVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)
        self.clf = None
        self.ok = False

    def _txt(self, lang, src):
        return "\x01" + lang + "\x02" + src

    def fit(self, pairs):
        # pairs: list of (lang, src, is_del)
        X = [self._txt(l, s) for l, s, _ in pairs]
        y = np.array([1 if d else 0 for _, _, d in pairs])
        if y.sum() < 5 or y.sum() == len(y):
            self.ok = False
            return self
        Xf = self.vec.fit_transform(X)
        self.clf = LogisticRegression(max_iter=400, class_weight="balanced", C=2.0)
        self.clf.fit(Xf, y)
        self.ok = True
        return self

    def p_del(self, lang, src):
        if not self.ok:
            return 0.0
        Xf = self.vec.transform([self._txt(lang, src)])
        return float(self.clf.predict_proba(Xf)[0, 1])

    def is_del(self, lang, src):
        return self.p_del(lang, src) >= self.thr
