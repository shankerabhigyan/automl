import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

class DataEater:
    def __init__(self, data_path, strategy=None, target_column=None):
        self.data_path = data_path
        self.strategy = strategy
        self.data = None
        self.target_column = target_column

    def fetch_data(self):
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path)
        # elif self.data_path.endswith('.json'):
        #     return pd.read_json(self.data_path)
        elif self.data_path.startswith('http'):
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError('Unsupported file format')
    
    def clean(self):
        if self.strategy=='mean':
            self.data.fillna(self.data.mean(), inplace=True)
        elif self.strategy=='median':
            self.data.fillna(self.data.median(), inplace=True)
        elif self.strategy=='mode':
            self.data.fillna(self.data.mode(), inplace=True)
        elif self.strategy==None:
            self.data.dropna(inplace=True)
        else:
            raise ValueError('Unsupported strategy for preprocessing')
        
class EDA:
    def __init__(self,data, target_column=None):
        self.data = data
        self.target_column = target_column

    def summarize(self):
        print(self.data.describe(include='all'))
        print(self.data.info())

    def plot_corr(self):
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def generate_column_distributions(self):
        fig, axes = plt.subplots(nrows=len(self.data.columns), ncols=1, figsize=(10, 10))
        for i, column in enumerate(self.data.columns):
            sns.histplot(
                self.data[column],  
                kde=True,
                stat='density',
                kde_kws=dict(cut=3),
                ax=axes[i]
            )
            axes[i].set_title(f'Distribution of {column}')
        plt.tight_layout()
        plt.show()

    def plot_pairwise(self):
        sns.pairplot(self.data)
        plt.title('Pairwise Plot')
        plt.show()

    def find_task_type(self):
        if self.target_column is not None:
            target_series = self.data[self.target_column]
            if pd.api.types.is_numeric_dtype(target_series):
                num_unique = target_series.nunique()
                total = len(target_series)
                if num_unique < min(10, total*0.05):
                    return 'classification'
                else:
                    return 'regression'
            else:
                return 'classification'
        else:
            return 'clustering'



