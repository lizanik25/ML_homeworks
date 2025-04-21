import numpy as np 

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        ind = np.random.choice(X.shape[0], size=(2, 1000), replace=False)
        ind = ind[:, ind[0] != ind[1]] 
        sigma = np.median(np.sum((X[ind[0]] - X[ind[1]]) ** 2, axis=1)) ** 0.5
        self.w = np.random.normal(0, 1 / sigma, size=(X.shape[1], self.n_features))
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        return self

    def transform(self, X, y=None):
        X_new = np.cos(X @ self.w + self.b)
        return X_new

class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def _compute_w(self, d):
        q, r = np.linalg.qr(np.random.normal(size=(d, d)))
        s = np.sqrt(np.random.chisquare(d, d))
        return 1 / self.sigma * (q @ np.diag(s))

    def fit(self, X, y=None):
        if self.n_features > self.new_dim and self.n_features % self.new_dim != 0:
            raise ValueError("n_features should be a multiple of new_dim when n_features > new_dim")
        
        ind = np.random.choice(X.shape[0], size=(2, 1000), replace=False)
        ind = ind[:, ind[0] != ind[1]]
        self.sigma = np.median(np.sum((X[ind[0]] - X[ind[1]]) ** 2, axis=1)) ** 0.5

        if self.n_features <= X.shape[1]:
            self.w = self._compute_w(X.shape[1])[:, :self.n_features]
        else:
            ws = []
            for _ in range(self.n_features // X.shape[1]):
                ws.append(self._compute_w(X.shape[1]))
            self.w = np.concatenate(ws, axis=1)
        
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        return self

    def transform(self, X, y=None):
        X_new = np.cos(X @ self.w + self.b)
        return X_new

class RFFPipeline(BaseEstimator):
    """
    Пайплайн, включающий:
    1. Опциональное применение PCA
    2. Преобразование признаков через RFF или ORF
    3. Классификацию (логистическая регрессия или линейный SVM)
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=RandomFeatureCreator,
            classifier_class='LogisticRegression',
            classifier_params=None,
            func=np.cos,
    ):
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.func = func

        if classifier_params is None:
            classifier_params = {}
        
        if isinstance(classifier_class, str):
            if classifier_class == 'LogisticRegression':
                self.classifier = LogisticRegression(max_iter=1000, **classifier_params)
            elif classifier_class == 'SVM':
                self.classifier = SVC(kernel="linear", **classifier_params)
            else:
                raise ValueError("Unsupported classifier. Choose 'LogisticRegression' or 'SVM'.")
        else:
            self.classifier = classifier_class(**classifier_params)

        self.feature_creator = feature_creator_class(n_features=self.n_features, new_dim=self.new_dim, func=func)
        self.pipeline = None

    def fit(self, X, y):
        steps = []
        if self.use_PCA:
            steps.append(('pca', PCA(n_components=self.new_dim)))
        steps.append(('feature_creator', self.feature_creator))
        steps.append(('classifier', self.classifier))
        
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)