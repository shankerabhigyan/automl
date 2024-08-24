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
            self.layers.append(nn.Tanh())
            for i in range(1, len(layers)):
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
                self.layers.append(nn.Tanh())
            self.layers.append(nn.Linear(layers[-1], output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) if isinstance(layer, nn.Linear) else layer(x)
        return x

    def configure_optim(self, lr=0.01, opt='Adam'):
        if opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer {opt}")

class evolveRegressionNN:
    def __init__(self, data_X, data_y):
        self.data_X = data_X
        self.data_y = data_y
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_int", random.randint, 10, 100)
        self.toolbox.register("individual", self.init_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)

    def init_individual(self, icls):
        num_layers = random.randint(1, 5)
        layers = [random.randint(10, 100) for _ in range(num_layers)]
        epochs = random.randint(50, 200)
        lr = random.choice([0.01, 0.001, 0.0001])
        optimizer = random.choice(['Adam', 'SGD'])
        return icls(layers + [epochs, lr, optimizer])


    def evaluate_individual(self, individual):
        *layers, epochs, lr, optimizer = individual
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
            loss.backward()
            optimizer.step()
        return loss.item(),

    def mutate_individual(self, individual, indpb=0.1):
        size = len(individual) - 3  # Avoid mutating the last three items (epochs, lr, optimizer)
        for i in range(size):
            if random.random() < indpb:
                individual[i] = random.randint(10, 100)
        if random.random() < indpb:
            individual[-3] = random.randint(50, 200)  # epochs
        if random.random() < indpb:
            individual[-2] = random.choice([0.01, 0.001, 0.0001])  # learning rate
        # Explicitly preserve the optimizer as a string and ensure no accidental type change
        individual[-1] = random.choice(['Adam', 'SGD'])  # optimizer, always a string
        return (individual,)

    def setup_toolbox(self):
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    datax = np.random.rand(100, 10)
    datay = np.random.rand(100, 1)
    eamodel = evolveRegressionNN(datax, datay)
    eamodel.setup_toolbox()
    pop = eamodel.toolbox.population(n=50)
    result = algorithms.eaSimple(pop, eamodel.toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=True)
    best_individual = tools.selBest(pop, 1)[0]
    print(f"Best individual: {best_individual} with fitness {best_individual.fitness.values}")
