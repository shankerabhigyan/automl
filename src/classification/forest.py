import torch
import numpy as np
from sklearn.utils import resample # for bootstrapping
from .tree import DecisionTree, Node

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, criterion='gini', max_features=None, bootstrap=True, device=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trees = []

    def _majority_vote(self,tree_preds):
        """
        Return the most common class label in the prediction
        """
        return torch.mode(tree_preds, dim=1).values
        
    
    def _bootstrap(self,x,y):
        """
        Create a bootstrap sample from x and y
        """
        indices = torch.randint(0, x.size(0), (x.size(0),), device=self.device)
        return x[indices], y[indices]
    
    def predict(self, x):
        """
        Predict class for x
        """
        x = x.to(self.device)
        tree_preds = torch.zeros((x.shape[0], len(self.trees)), dtype=torch.long, device=self.device)

        for i, (tree, feature_indices) in enumerate(self.trees):
            if feature_indices is not None:
                x_subset = x[:, feature_indices]
            else:
                x_subset = x
            tree_preds[:, i] = tree.predict(x_subset)

        majority_vote =  self._majority_vote(tree_preds)
        return majority_vote
    
    def fit(self, x, y):
        """
        fit to data..
        """
        # send data to device
        x, y = x.to(self.device), y.to(self.device)
        n_samples, n_features = x.size()
        
        for _ in range(self.n_estimators):
            if self.bootstrap:
                x_sample, y_sample = self._bootstrap(x, y)
            else:
                x_sample, y_sample = x, y

            feature_indices = None
            if self.max_features is not None:
                if self.max_features == 'sqrt':
                    self.max_features = int(np.sqrt(n_features))
                elif self.max_features == 'log2':
                    self.max_features = int(np.log2(n_features))
                feature_indices = np.random.choice(n_features, self.max_features, replace=False) # select random features
                x_sample = x_sample[:, feature_indices]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion=self.criterion, device=self.device)
            tree.fit(x_sample, y_sample)
            self.trees.append((tree, feature_indices))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    
    X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    forest = RandomForest(n_estimators=100, max_depth=5, max_features=5, bootstrap=True, criterion='gini', device=device)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)

    y_pred = y_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()