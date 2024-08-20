import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

"""
TODO : 
- [x] Implement the proxy NN class
- [x] Implement the SBE class
- [ ] Implement LR scheduler
- [ ] Implement early stopping
"""

class proxyNN(nn.Module):
    """
    Proxy NN to be used for feature selection.
    we keep the number of parameters to be 30 to 70% of the number of samples
    """
    def __init__(self, input_dim, n_samples, output_dim=1):
        super(proxyNN, self).__init__() 
        self.hidden_dim = int(n_samples*0.5)
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.fc3 = nn.Linear(self.hidden_dim//2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x
    

class SBE:
    """
    Sequential Backward Elimination for feature selection.
    """
    def __init__(self, lossfn=nn.MSELoss(), lr=0.01, epochs=10):
        self.lossfn = lossfn
        self.lr = lr
        self.epochs = epochs
        self.model = None
        # optimizer to be SGD with nesterov accelerated gradients
        # Kingma et. al. 2014, Zhou et. al. 2020
        self.optimizer = None

    def _train_model(self, xtrain, ytrain):
        n_samples = xtrain.shape[0]
        output_dim = 1 # regression task
        input_dim = xtrain.shape[1]
        self.model = proxyNN(input_dim, n_samples, output_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, nesterov=True, momentum=0.9, dampening=0)
        x_tensor = torch.tensor(xtrain, dtype=torch.float32)
        y_tensor = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(x_tensor)
            loss = self.lossfn(y_pred, y_tensor)
            loss.backward()
            self.optimizer.step()
            print(f'SBE => Epoch: {epoch+1}/{self.epochs} || Loss: {loss.item()}')

        return self.model
    
    def _evaluate_model(self, xtest, ytest, model):
        x_tensor = torch.tensor(xtest, dtype=torch.float32)
        y_tensor = torch.tensor(ytest, dtype=torch.float32).view(-1, 1)
        model.eval()
        y_pred = model(x_tensor)
        loss = self.lossfn(y_pred, y_tensor)
        print("\nSBE => Evaluation Loss: {}\n".format(loss.item()))
        return loss.item()
    
    def sequentialBackwardElimination(self, X, y):
        remaining_features = list(range(X.shape[1]))
        best_features = remaining_features[:]

        xtrain, xtest, ytrain, ytest = train_test_split(X[:, best_features], y, test_size=0.2)
        model = self._train_model(xtrain, ytrain)
        best_loss = self._evaluate_model(xtest, ytest, model)

        while len(remaining_features)>0:
            scores = []
            for feature in remaining_features:
                features_to_test = [f for f in remaining_features if f != feature]
                xtrain, xtest, ytrain, ytest = train_test_split(X[:,features_to_test], y, test_size=0.2)
                model = self._train_model(xtrain, ytrain)
                loss = self._evaluate_model(xtest, ytest, model)
                scores.append((feature, loss))

            scores = sorted(scores, key=lambda x: x[1])
            best_score, worst_feature = scores[0]

            # evaluate model on all best features
            
            if best_score >= best_loss:
                break # no improvement

            remaining_features.remove(worst_feature)
            best_features = remaining_features[:]
            best_loss = best_score

        return best_features


