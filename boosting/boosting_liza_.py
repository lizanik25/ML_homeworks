from __future__ import annotations
from collections import defaultdict
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

def score(clf, x, y):
    return roc_auc_score((y == 1), clf.predict_proba(x)[:, 1])

class Boosting:
    def __init__(
        self,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        subsample: float = 0.3,
        early_stopping_rounds: int = None,
        plot: bool = False,
        bootstrap_type: str = 'bernoulli',
        rsm: float = 1.0,
        quantization_type: Optional[str] = None,
        nbins: int = 255,
        bagging_temperature: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        subsample_goss: float = 0.3,
        dart: bool = False,    
        dropout_rate: float = 0.05,  
    ):
        if base_model_params is None:
            base_model_params = {'max_depth': 3, 'min_samples_leaf': 5}
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params = base_model_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.plot = plot
        self.bootstrap_type = bootstrap_type.lower()
        self.rsm = rsm
        self.quantization_type = quantization_type.lower() if quantization_type else None
        self.nbins = nbins
        self.bagging_temperature = bagging_temperature
        self.goss = goss
        self.goss_k = goss_k
        self.subsample_goss = subsample_goss
        self.dart = dart               
        self.dropout_rate = dropout_rate
        self.models: list = []
        self.gammas: list = []
        self.history = defaultdict(list)
        self.best_loss_ = np.inf
        self.patience_cnt_ = 0
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def _bootstrap_sample(self, X, y):
        if self.bootstrap_type == 'bernoulli':
            mask = np.random.binomial(n=1, p=self.subsample, size=X.shape[0]).astype(bool)
            return X[mask], y[mask]
        elif self.bootstrap_type == 'subsample':
            size = int(self.subsample * X.shape[0])
            perm = np.random.permutation(X.shape[0])
            idx = perm[:size]
            return X[idx], y[idx]
        elif self.bootstrap_type == 'bayesian':
            u = np.random.rand(X.shape[0])
            w = (-np.log(u)) ** self.bagging_temperature
            s = w.sum()
            p = w / s if s > 0 else np.ones_like(w)/ len(w)
            idx = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True, p=p)
            return X[idx], y[idx]
        else:
            raise ValueError("Unsupported bootstrap_type. Use 'bernoulli', 'subsample', or 'bayesian'.")

    def _apply_goss(self, X, residuals):
        total = len(residuals)
        sorted_idx = np.argsort(np.abs(residuals))[::-1]
        large_count = int(self.goss_k * total)
        large_idx = sorted_idx[:large_count]
        small_idx = sorted_idx[large_count:]
        small_count = int(self.subsample_goss * len(small_idx))
        if small_count > 0:
            small_sample_idx = np.random.choice(small_idx, small_count, replace=False)
        else:
            small_sample_idx = np.array([], dtype=int)
        sel_idx = np.concatenate([large_idx, small_sample_idx])
        Xb = X[sel_idx]
        rb = residuals[sel_idx].copy()
        if small_count == 0:
            swf = 1.0
        else:
            swf = len(small_idx) / float(small_count)
        rb[:len(large_idx)] *= 1.0
        rb[len(large_idx):] *= swf
        return Xb, rb, sel_idx

    def _quantize_col_uniform(self, col_vals):
        mn, mx = col_vals.min(), col_vals.max()
        if mx > mn:
            bins = np.linspace(mn, mx, self.nbins + 1)
            return np.digitize(col_vals, bins) - 1
        return np.zeros_like(col_vals)

    def _quantize_col_quantile(self, col_vals):
        uq = np.unique(col_vals)
        if len(uq) <= 1:
            return np.zeros_like(col_vals)
        qtls = np.percentile(col_vals, np.linspace(0, 100, self.nbins + 1))
        return np.digitize(col_vals, qtls) - 1

    def _quantize(self, X, y_vals=None):
        if not self.quantization_type or self.quantization_type not in ['uniform', 'quantile']:
            return X
        Xq = X.copy()
        for c in range(Xq.shape[1]):
            if self.quantization_type == 'uniform':
                Xq[:, c] = self._quantize_col_uniform(Xq[:, c])
            elif self.quantization_type == 'quantile':
                Xq[:, c] = self._quantize_col_quantile(Xq[:, c])
        return Xq

    def find_optimal_gamma(self, y, old_pred, new_pred):
        gs = np.linspace(0, 1, 100)
        losses = [self.loss_fn(y, old_pred + g * new_pred) for g in gs]
        return gs[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def _randomize_features(self, X):
        n_features = X.shape[1]
        if isinstance(self.rsm, float):
            k = int(self.rsm * n_features)
        else:
            k = int(self.rsm)
        k = max(1, min(k, n_features))
        feats = np.random.choice(n_features, size=k, replace=False)
        return X[:, feats], feats

    def partial_fit(self, X, y, pred):
        if self.dart and len(self.models) > 0:
            nt = len(self.models)
            n_drop = int(nt * self.dropout_rate)
            if n_drop > 0:
                drop_idx = np.random.choice(nt, n_drop, replace=False)
                scale_factor = nt / (nt - n_drop)
                for idx in drop_idx:
                    preds_to_remove = self.learning_rate * self.gammas[idx] * self.models[idx][0].predict(X[:, self.models[idx][1]])
                    pred -= preds_to_remove
                pred *= scale_factor

        res = -self.loss_derivative(y, pred)
        if self.goss:
            Xb, rb, _ = self._apply_goss(X, res)
        else:
            Xb, rb = self._bootstrap_sample(X, res)
        Xr, feats = self._randomize_features(Xb)
        Xr = self._quantize(Xr, None)
        mdl = self.base_model_class(**self.base_model_params)
        mdl.fit(Xr, rb)
        Xall = self._quantize(X[:, feats], None)
        nd = mdl.predict(Xall)
        g_opt = self.find_optimal_gamma(y, pred, nd)
        if self.dart and len(self.models) > 0 and n_drop > 0:
            g_opt *= scale_factor
        self.models.append((mdl, feats))
        self.gammas.append(g_opt)
        return g_opt, mdl, feats

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        yt = (2 * y_train - 1)
        ptr = np.zeros_like(yt, dtype=float)
        pvr = None
        yv_ = None
        if X_valid is not None and y_valid is not None:
            yv_ = (2 * y_valid - 1)
            pvr = np.zeros_like(yv_, dtype=float)
        for i in range(self.n_estimators):
            g_opt, mdl, feats = self.partial_fit(X_train, yt, ptr)
            Xt = self._quantize(X_train[:, feats], None)
            prt = mdl.predict(Xt)
            ptr += self.learning_rate * g_opt * prt
            lt = self.loss_fn(yt, ptr)
            self.history['train_loss'].append(lt)
            at = self.score(X_train, y_train)
            self.history['train_auc'].append(at)
            if X_valid is not None and y_valid is not None:
                Xv = self._quantize(X_valid[:, feats], None)
                prv = mdl.predict(Xv)
                pvr += self.learning_rate * g_opt * prv
                lv = self.loss_fn(yv_, pvr)
                self.history['valid_loss'].append(lv)
                av = self.score(X_valid, y_valid)
                self.history['valid_auc'].append(av)
                if lv < self.best_loss_:
                    self.best_loss_ = lv
                    self.patience_cnt_ = 0
                else:
                    self.patience_cnt_ += 1
                if self.early_stopping_rounds and self.patience_cnt_ >= self.early_stopping_rounds:
                    for _ in range(i + 1, self.n_estimators):
                        self.models.append((mdl, feats))
                        self.gammas.append(0.0)
                    break
        if len(self.models) < self.n_estimators:
            diff = self.n_estimators - len(self.models)
            for _ in range(diff):
                self.models.append((mdl, feats))
                self.gammas.append(0.0)

    def predict_proba(self, X):
        s = np.zeros(X.shape[0], dtype=float)
        for (mdl, feats), gm in zip(self.models, self.gammas):
            Xr = self._quantize(X[:, feats], None)
            s += self.learning_rate * gm * mdl.predict(Xr)
        pr = self.sigmoid(s)
        return np.column_stack((1 - pr, pr))

    def plot_history(self, X=None, y=None):
        ftr, ax_tr = plt.subplots(1, 2, figsize=(12, 5))
        if 'train_loss' in self.history:
            ax_tr[0].plot(self.history['train_loss'], label='train_loss')
        ax_tr[0].set_title('Train Loss')
        ax_tr[0].set_xlabel('Iteration')
        ax_tr[0].set_ylabel('Loss')
        ax_tr[0].legend()
        if 'train_auc' in self.history:
            ax_tr[1].plot(self.history['train_auc'], label='train_auc')
        ax_tr[1].set_title('Train AUC')
        ax_tr[1].set_xlabel('Iteration')
        ax_tr[1].set_ylabel('AUC')
        ax_tr[1].legend()
        plt.tight_layout()
        plt.show()
        if 'valid_loss' in self.history or 'valid_auc' in self.history:
            fvl, ax_vl = plt.subplots(1, 2, figsize=(12, 5))
            if 'valid_loss' in self.history:
                ax_vl[0].plot(self.history['valid_loss'], label='valid_loss')
            ax_vl[0].set_title('Validation Loss')
            ax_vl[0].set_xlabel('Iteration')
            ax_vl[0].set_ylabel('Loss')
            ax_vl[0].legend()
            if 'valid_auc' in self.history:
                ax_vl[1].plot(self.history['valid_auc'], label='valid_auc')
            ax_vl[1].set_title('Validation AUC')
            ax_vl[1].set_xlabel('Iteration')
            ax_vl[1].set_ylabel('AUC')
            ax_vl[1].legend()
            plt.tight_layout()
            plt.show()

    def feature_importances_(self):
        if not self.models:
            return None
        mx = max(feat.max() for (_, feat) in self.models if len(feat) > 0) + 1
        total = np.zeros(mx, dtype=float)
        for (mdl, feats) in self.models:
            if hasattr(mdl, 'feature_importances_'):
                for i, c in enumerate(feats):
                    total[c] += mdl.feature_importances_[i]
        s = total.sum()
        return total / s if s > 0 else total
