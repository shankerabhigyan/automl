import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms
import numpy as np
import random

class BaseNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        super(BaseNN, self).__init__()
        self.layers = nn.ModuleList()
        if len(layers) > 0:
            self.layers.append(nn.Linear(input_dim, layers[0]))
            self.layers.append(nn.BatchNorm1d(layers[0]))  # Add BatchNorm
            self.layers.append(nn.ReLU())
            for i in range(1, len(layers)):
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
                self.layers.append(nn.BatchNorm1d(layers[i]))  # Add BatchNorm
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(layers[-1], output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def configure_optim(self, lr=0.01, opt='Adam'):
        if opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        elif opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, weight_decay=1e-5)
        else:
            raise ValueError(f"Unsupported optimizer {opt}")

class evolveRegressionNN:
    def __init__(self, data_X, data_y, val_X, val_y):
        self.data_X = data_X
        self.data_y = data_y
        self.val_X = val_X
        self.val_y = val_y

    def init_individual(self, icls):
        num_layers = random.randint(1, 5)
        layers = [random.randint(500, 1000) for _ in range(num_layers)]
        epochs = random.randint(500, 3000)
        lr = random.choice([0.01, 0.001, 0.0001])
        optimizer = random.choice(['Adam', 'SGD'])
        return icls({'layers': layers, 'epochs': epochs, 'lr': lr, 'optimizer': optimizer})

    def evaluate_individual(self, individual):
        layers = individual['layers']
        epochs = individual['epochs']
        lr = individual['lr']
        optimizer = individual['optimizer']

        if not isinstance(optimizer, str):
            raise ValueError(f"Optimizer type must be a string, got {type(optimizer)} instead.")
        print(f"Initialising model with layers : {layers}, epochs :{epochs}, lr : {lr} and optimizer : {optimizer}")
        
        model = BaseNN(input_dim=self.data_X.shape[1], output_dim=1, layers=layers)
        optimizer = model.configure_optim(lr=lr, opt=optimizer)
        criterion = nn.MSELoss()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        X_tensor = torch.tensor(self.data_X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(self.data_y, dtype=torch.float32).view(-1, 1).to(device)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            if torch.isnan(loss).any():
                print(f"Loss is NaN at epoch {epoch}, skipping")
                continue
            loss.backward()
            optimizer.step()
            # if epoch % 500 == 0:
            #     print(f"Epoch {epoch + 1} => Loss: {loss.item()}")
        print(f"Training loss: {loss.item()}")
        with torch.no_grad():
            model.eval()
            val_X_tensor = torch.tensor(self.val_X, dtype=torch.float32).to(device)
            val_y_tensor = torch.tensor(self.val_y, dtype=torch.float32).view(-1, 1).to(device)
            val_outputs = model(val_X_tensor)
            val_loss = criterion(val_outputs, val_y_tensor)
            print(f"Validation loss: {val_loss.item()}")
            # % loss
            mape = torch.mean(torch.abs((val_outputs - val_y_tensor) / val_y_tensor)) * 100
            print(f"Validation MAPE: {mape.item()}")
        return 1/val_loss.item(), 

    def mutate_individual(self, individual, indpb=0.1):
        # Mutate layer sizes
        if random.random() < indpb:
            num_layers = random.randint(1, 5)
            individual['layers'] = [random.randint(100, 1000) for _ in range(num_layers)]

        # Mutate epochs
        if random.random() < indpb:
            individual['epochs'] = random.randint(500, 10000)

        # Mutate learning rate
        if random.random() < indpb:
            individual['lr'] = random.choice([0.01, 0.001, 0.0001])

        # Mutate optimizer
        if random.random() < indpb:
            individual['optimizer'] = random.choice(['Adam', 'SGD'])

        return (individual,)
    
    def crossover(self, ind1, ind2):
        for key in ind1.keys():
            if random.random() < 0.5:
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    def setup_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.init_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evolve(self):
        self.setup_toolbox()
        pop = self.toolbox.population(n=50)
        result = algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.4, ngen=40, verbose=True)
        best_individual = tools.selBest(pop, 1)[0]
        print(f"Best individual: {best_individual} with fitness {best_individual.fitness.values}")
        return best_individual

if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    # split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    eamodel = evolveRegressionNN(X_train, y_train, X_test, y_test)
    eamodel.setup_toolbox()
    pop = eamodel.toolbox.population(n=50)
    result = algorithms.eaSimple(pop, eamodel.toolbox, cxpb=0.7, mutpb=0.4, ngen=40, verbose=True)
    best_individual = tools.selBest(pop, 1)[0]
    print(f"Best individual: {best_individual} with fitness {best_individual.fitness.values}")

