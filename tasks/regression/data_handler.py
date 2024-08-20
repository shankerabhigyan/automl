import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.impute import SimpleImputer 

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