import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.impute import SimpleImputer 
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class DataCleaner:
    """
    To clean and preprocess data for regression task.
    Takes a pandas DataFrame as input(for now).
    """
    def __init__(self, strategy='mean', data=None, outlier_threshold=3.0):
        self.strategy = strategy
        self.outlier_threshold = outlier_threshold
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    def _remove_duplicates(self, data):
        return data.drop_duplicates()

    def _handle_missing_values(self, data):
        numeric_cols = data.select_dtypes(include=['int64','float64']).columns
        data_numeric = data[numeric_cols]
        data_imputed = pd.DataFrame(self.imputer.fit_transform(data_numeric), columns=data_numeric.columns)

        non_numeric_cols = data.select_dtypes(exclude=['int64','float64']).columns
        if len(non_numeric_cols) > 0:
            data_non_numeric = data[non_numeric_cols].reset_index(drop=True)
            data_imputed = pd.concat([data_imputed, data_non_numeric], axis=1)
        return data_imputed

    def _handle_outliers(self,data):
        z_scores = np.abs(stats.zscore(data))
        data = data[(z_scores < self.outlier_threshold).all(axis=1)]
        return data

    def _handle_categorical_features(self, data):
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) > 0:
            encoded = pd.DataFrame(self.encoder.fit_transform(data[cat_cols]), columns=self.encoder.get_feature_names_out(cat_cols))
            data = data.drop(cat_cols, axis=1)  # Assign back to data
            data = pd.concat([data, encoded], axis=1)
        
        return data


    def _normalize_data(self,data):
        return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
    
    def clean(self, data):
        data = self._remove_duplicates(data)
        data = self._handle_missing_values(data)
        data = self._handle_outliers(data)
        data = self._handle_categorical_features(data)
        data = self._normalize_data(data)
        return data
    

class DataSplitter:
    """
    To split data into train, test, and validation sets.
    We use KFold for splitting since this is a regression task.
    """
    def __init__(self, n_splits=3, random_state=42, type='kfold'):
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.type = type
    
    def kfold_split(self, data, target):
        # train, test and validation sets
        for train_index, test_index in self.kf.split(data):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.kf.random_state)
        return X_train, X_test, X_val, y_train, y_test, y_val


class tSNE:
    """
    take in regression data and perform tSNE on it and return the transformed, reduced data.
    """
    def __init__(self, data, perplexity=30, n_components=2, lr=200.0, n_iter=1000, rs=None):
        self.data = data
        self.perplexity = perplexity
        self.n_components = n_components
        self.lr = lr
        self.n_iter = n_iter
        self.rs = rs
        if self.rs is not None:
            np.random.seed(self.rs)
    
    def _compute_pairwise_affinities(self,X):
        """
        compute pairwise affinities in high dim space.
        return pairwise affinity matrix p_ij
        x : data (n_samples, n_features)
        """
        (n_samples, n_features) = X.shape
        P = np.zeros((n_samples, n_samples))
        sigma_squared = np.ones(n_samples)

        for i in range(n_samples):
            sum_xi = np.sum((X[i]-X)**2, axis=1) # axis = 1 for row-wise sum
            beta = 1.0 / (2.0 * sigma_squared[i])
            min_beta = -np.inf
            max_beta = np.inf
            tolerance = 1e-5 # tolerance for binary search convergence
            target_perplexity = np.log(self.perplexity) # can be compared directly with H_i

            for _ in range(50):
                P_i = np.exp(-beta * sum_xi)
                P_i[i] = 0 # exclude self-affinity(not considered in cnd-probability calculation)
                sum_P_i = np.sum(P_i)
                if sum_P_i == 0:
                    sum_P_i = 1 # avoid division by zero
                H_i = np.log(sum_P_i) + beta * np.sum(sum_xi * P_i) / sum_P_i
                diff = H_i - target_perplexity
                if np.abs(diff) <= tolerance:
                    break
                if diff > 0: 
                # this means that the points are too spread out, we need to increase beta
                # note that higher beta means a narrower gaussian while H_i is the entropy of the gaussian
                # higher H_i means the gaussian is more spread out
                # so if H_i is higher than target_perplexity, we need to increase beta
                    min_beta = beta # move beta to the right
                    if max_beta == np.inf:
                        beta = beta * 2
                    else:
                        beta = (beta + max_beta) / 2
                else: # beta is too high, we need to decrease it
                    max_beta = beta
                    if min_beta == -np.inf:
                        beta = beta / 2
                    else:
                        beta = (beta + min_beta) / 2

            P[i,:] = P_i / sum_P_i
            sigma_squared[i] = 1 / (2 * beta)
        P = (P + P.T) / (2 * n_samples) # symmetrize P
        return P
    
    def _compute_low_dimensional_affinities(self, Y):
        """
        
        """