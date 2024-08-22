import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class proxyNN(nn.Module):
    """
    Proxy Neural Network to be used for feature selection.
    The number of parameters is kept between 30% to 70% of the number of samples.
    """
    def __init__(self, input_dim, n_samples, output_dim=1, device='cpu'):
        super(proxyNN, self).__init__()
        self.device = device
        self.hidden_dim = int(n_samples * 0.5)
        self.fc1 = nn.Linear(input_dim, self.hidden_dim).to(device)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2).to(device)
        self.fc3 = nn.Linear(self.hidden_dim // 2, output_dim).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x

class SBE:
    """
    Sequential Backward Elimination (SBE) for feature selection.
    """
    def __init__(self, lossfn=nn.MSELoss(), lr=0.01, epochs=10, device='cpu'):
        self.device = device
        self.lossfn = lossfn
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.optimizer = None

    def _train_model(self, xtrain, ytrain):
        n_samples = xtrain.shape[0]
        output_dim = 1  # For regression task
        input_dim = xtrain.shape[1]
        self.model = proxyNN(input_dim, n_samples, output_dim, self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, nesterov=True, momentum=0.9, dampening=0)

        x_tensor = torch.tensor(xtrain, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(x_tensor)
            loss = self.lossfn(y_pred, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1} => Loss: {loss.item()}")

        return self.model

    def _evaluate_model(self, xtest, ytest, model):
        x_tensor = torch.tensor(xtest, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(ytest, dtype=torch.float32).view(-1, 1).to(self.device)
        model.eval()
        with torch.no_grad():
            y_pred = model(x_tensor)
            loss = self.lossfn(y_pred, y_tensor)
        print("\nSBE => Evaluation Loss: {}\n".format(loss.item()))
        return loss.item()

    def sequentialBackwardElimination(self, X, y):
        remaining_features = list(range(X.shape[1]))
        best_features = remaining_features[:]
        xtrain, xtest, ytrain, ytest = train_test_split(X[:, best_features], y, test_size=0.2)
        print("Starting SBE with features {}".format(best_features))
        model = self._train_model(xtrain, ytrain)
        best_loss = self._evaluate_model(xtest, ytest, model)
        print(f"SBE => Initial loss: {best_loss}")

        while len(remaining_features) > 0:
            scores = []
            for feature in remaining_features:
                print(f"SBE => Testing relevance for {feature}")
                features_to_test = [f for f in remaining_features if f != feature]
                xtrain_ = xtrain[:, features_to_test]
                xtest_ = xtest[:, features_to_test]
                model = self._train_model(xtrain_, ytrain)
                loss = self._evaluate_model(xtest_, ytest, model)
                scores.append((loss, feature))

            scores = sorted(scores, key=lambda x: x[1])
            best_score, worst_feature = scores[-1]
            print(f"SBE => Best score: {best_score}, Worst feature: {worst_feature}")

            if best_score >= best_loss:
                print("SBE => Stopping, no improvement")
                break
            print(f"SBE => Removing feature {worst_feature} with loss {best_score}")
            remaining_features.remove(worst_feature)
            best_features = remaining_features[:]
            xtrain = xtrain[:, best_features]
            xtest = xtest[:, best_features]
            best_loss = best_score

        return best_features