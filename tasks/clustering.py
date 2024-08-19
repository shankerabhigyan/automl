"""
OVERVIEW:
    - *Implementing SFS(Sequential Forward Search) for feature selection
    - *Implementing EM Clustering with Gaussian groups assumption
    - We allow the number of clusters to adjust to the data
    - *Implementing Feature Subset Evaluation Criteria
    - We use two criterion:
        - Scatter Separability Criterion(feature subset evaluation) [trace(Sb*Sw^(-1))]
        - Maximum Likelihood Criterion(clustering evaluation)
"""


import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_random_state 


class DFTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that takes a pandas DataFrame and returns a numpy array.
    For the sake of simplicity, we will assume that the DataFrame contains only numerical columns.
    TODO - Implement a more general version that can handle categorical columns as well.
    """
    def __init__(self, df, scale=True):
        self.df = df
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        if self.scale:
            self.scaler.fit(X)
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'scaler')
        X = self.scaler.transform(X)
        return X
    
    def fit_transform(self, X, y=None):
        if self.scale:
            X = self.scaler.fit_transform(X)
        return X


class SFS(BaseEstimator, TransformerMixin):
    """
    Sequential Forward Search for feature selection
    """
    def __init__(self, criterion='scatter', min_clusters=2, max_clusters=10, random_state=None):
        self.random_state = random_state
        self.criterion = criterion
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.selected_features = []
        self.tolerance = 1e-4

    def _normalize(self, X):
        """
        Normalize the data
        """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    def _objective_function(self, X, gmm):
        log_likelihood = gmm.score(X) * len(X)
        n_params = gmm._n_parameters()
        N = X.shape[0]
        F_k = log_likelihood - 0.5*n_params*np.log(N)
        return F_k
    
    def _merge_clusters(self, gmm, X):
        """
        we merge the two clusters which result in the min difference in the objective function
        """
        min_delta_F = np.inf
        best_merge = None 
        best_gmm = None # best gmm after merging
        for i in range(gmm.n_components):
            for j in range(i+1, gmm.n_components):
                new_gmm = GaussianMixture()

    def _em_clustering(self, X):
        """
        we use the gauusian mixture model to cluster the data
        and dynamically adjust the number of clusters
        """
        best_gmm = None
        best_bic = np.inf
        best_labels = None

        for n_clusters in range(self.min_clusters, self.max_clusters+1):
            gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
            labels = gmm.fit_predict(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_labels = labels
        return best_gmm, best_labels
    
    def _calculate_scatter_matrices(self, X, labels):
        """
        Calculate the scatter matrices Sb and Sw
        Sb = sum(n_i * (mu_i - mu) * (mu_i - mu)^T) -> between-class scatter matrix
        Sw = sum((x_i - mu_i) * (x_i - mu_i)^T) -> within-class scatter matrix
        """
        _, n_features = X.shape
        unique_labels = np.unique(labels)
        means = np.mean(X, axis=0)

        Sb = np.zeros((n_features, n_features))
        Sw = np.zeros((n_features, n_features))

        for label in unique_labels:
            x_class = X[labels == label]
            mean_class = np.mean(x_class, axis=0)
            Sw += np.dot((x_class - mean_class).T, (x_class - mean_class))
            n_class = x_class.shape[0]
            mean_diff = (mean_class - means).reshape(-1, 1)
            Sb += n_class * np.dot(mean_diff, mean_diff.T)
        return Sb, Sw
    
    def _scatter_separability_criterion(self, X):
        """
        Calculate the scatter separability criterion
        trace(Sb * Sw^(-1))
        """
        _, labels = self._em_clustering(X)
        Sb, Sw = self._calculate_scatter_matrices(X, labels)
        Sw_inv = inv(Sw)
        criterion_value = np.trace(np.dot(Sb, Sw_inv))
        return criterion_value
    
    def fit(self, X):
        _, n_features = X.shape
        available_features = list(range(n_features))
        best_criterion_value = -np.inf
        while available_features:
            best_feature = None
            for feature in available_features:
                candidate_features = self.selected_features + [feature]
                X_selected = X[:, candidate_features]
                # only taking scatter separability criterion for now
                criterion_value = self._scatter_separability_criterion(X_selected)
                if criterion_value > best_criterion_value + self.tolerance:
                    best_criterion_value = criterion_value
                    best_feature = feature
            if best_feature is not None:
                self.selected_features.append(best_feature)
                available_features.remove(best_feature)
            else:
                break
        return self.selected_features


if __name__ == '__main__':
    print("testing SFS...\n\n")
    ############
    # # dummy data for testing DFTransformer
    # data = np.random.rand(100, 5) # 100 samples, 5 features
    # df = pd.DataFrame(data, columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    # transformer = DFTransformer(df)
    # transformed_data = transformer.fit_transform(df)
    # print(transformed_data)
    ############
    # sfs = SFS(min_clusters=2, max_clusters=10)
    # X = np.random.rand(1000, 15)  # Dummy data: 100 samples, 10 features
    # X_normalized = sfs._normalize(X)
    # selected_features = sfs.fit(X_normalized)
    # print("Selected Features:", selected_features)
    ############
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data.data
    y = data.target
    dft = DFTransformer(df=pd.DataFrame(X))
    X = dft.fit_transform(X)
    sfs = SFS(min_clusters=2, max_clusters=10)
    selected_features = sfs.fit(X)
    print("Total Features:", X.shape[1])
    print("Total Feature Names:", data.feature_names)
    print("Selected Features:", selected_features)
    print("Selected Features:", [data.feature_names[i] for i in selected_features])
    