# decision tree classifier
import numpy as np
import torch

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini', device=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _predict_class(self, x, node):
        """
        Predict class for a single sample.
            - If node is a leaf node, return the value of the node
            - Otherwise, split the sample at the node's feature and threshold
            - Recursively call _predict_class on the left or right child until a leaf node is reached
        """
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_class(x, node.left)
        else:
            return self._predict_class(x, node.right)
        
    def _most_common_label(self, y):
        """
        Return the most common label in y
        """
        y = y.long()
        return torch.bincount(y).argmax().item()
    
    def _impurity(self, y):
        """
        Calculate impurity of y
        """
        y = y.long()
        class_count = torch.bincount(y)
        probabilities = class_count.float() / y.size(0)

        if self.criterion == 'gini':
            return 1-torch.sum(probabilities**2)
        elif self.criterion == 'entropy':
            return -torch.sum(probabilities * torch.log2(probabilities+1e-10)) # add small value to avoid log(0)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
    def _calculate_impurity(self, x, y, feature_index, threshold):
        """
        Calculate impurity of a split.
        feature_index: index of feature to split on
        threshold: threshold value for the split
        """
        left_mask = x[:, feature_index] < threshold
        right_mask = x[:, feature_index] >= threshold
        
        if(len(y[left_mask]) == 0 or len(y[right_mask]) == 0):
            return float('inf')
        
        left_impurity = self._impurity(y[left_mask])
        right_impurity = self._impurity(y[right_mask])

        weighted_impurity = (len(y[left_mask]) * left_impurity + len(y[right_mask]) * right_impurity) / len(y)
        return weighted_impurity

    def _find_best_split(self, x, y, num_features):
        """
        Find the best split for node.
        """
        best_feature, best_threshold, best_impurity = None, None, float('inf')

        for i in range(num_features):
            thresholds = torch.unique(x[:, i])
            for threshold in thresholds:
                impurity = self._calculate_impurity(x, y, i, threshold)
                if impurity < best_impurity:
                    best_feature, best_threshold, best_impurity = i, threshold, impurity

        return best_feature, best_threshold, best_impurity
    
    def _build_tree(self, x, y, depth):
        """
        Recursively build the tree.
            - If stopping criteria is met, return a leaf node
            - Otherwise, find the best split and recursively build the left and right child nodes
        """
        num_samples, num_features = x.shape
        unique_classes = torch.unique(y)

        if len(unique_classes)==1 or num_samples<self.min_samples_split or (self.max_depth is not None and depth>=self.max_depth):
            return Node(value=self._most_common_label(y)) # return leaf node
        
        best_feature, best_threshold, best_impurity = self._find_best_split(x, y, num_features)
        # make sure type is tensor
        if best_threshold is None:
            return Node(value=self._most_common_label(y)) # return leaf node if no split is found
        left_mask = x[:, best_feature] < best_threshold
        right_mask = x[:, best_feature] >= best_threshold
        left_subtree = self._build_tree(x[left_mask], y[left_mask], depth+1)
        right_subtree = self._build_tree(x[right_mask], y[right_mask], depth+1)

        return Node(best_feature, best_threshold, left_subtree, right_subtree)
    
    def predict(self, x):
        """
        Predict class for x.
        """
        x = torch.tensor(x, device=self.device)
        return torch.tensor([self._predict_class(sample, self.root) for sample in x], device=self.device)
    
    def fit(self, x, y):
        x, y = x.clone().detach(), y.clone().detach()
        self.root = self._build_tree(x, y, depth=0)

if __name__=="__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, y_train = torch.tensor(x_train, device=device), torch.tensor(y_train, device=device)
    x_test, y_test = torch.tensor(x_test, device=device), torch.tensor(y_test, device=device)

    model = DecisionTree(max_depth=5, device=device, criterion='gini')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())}")
    