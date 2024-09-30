import torch
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from .forest import RandomForest
import random

class evolveRandomForest:
    def __init__(self, x, y, val_x, val_y, device=None):
        self.x = x
        self.y = y
        self.val_x = val_x
        self.val_y = val_y
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_individual(self, icls):
        n_estimators = random.randint(10, 200)
        max_depth = random.choice([None, *range(1,20)])
        min_samples_split = random.randint(2, 10)
        criterion = random.choice(['gini', 'entropy'])
        max_features = random.choice(['sqrt', 'log2', None])
        bootstrap = random.choice([True, False])

        return icls({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'criterion': criterion,
            'max_features': max_features,
            'bootstrap': bootstrap
        })
    
    def _evaluate_individual(self, individual):
        n_estimators = individual['n_estimators']
        max_depth = individual['max_depth']
        min_samples_split = individual['min_samples_split']
        criterion = individual['criterion']
        max_features = individual['max_features']
        bootstrap = individual['bootstrap']

        rf = RandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            max_features=max_features,
            bootstrap=bootstrap,
            device=self.device
        )
        print(f"Fitting and Evaluiating chromosome {individual}")
        rf.fit(self.x, self.y)
        preds = rf.predict(self.val_x)
        print(f"Validation accuracy: {accuracy_score(self.val_y.cpu().numpy(), preds.cpu().numpy())}")
        acc =  accuracy_score(self.val_y.cpu().numpy(), preds.cpu().numpy()),
        return acc

    def _mutate_individual(self, individual, indpb=0.1):
        if random.random() < indpb:
            individual['n_estimators'] = random.randint(10, 200)
        
        if random.random() < indpb:
            individual['max_depth'] = random.choice([None, *range(1,20)])
        
        if random.random() < indpb:
            individual['min_samples_split'] = random.randint(2, 10)
        
        if random.random() < indpb:
            individual['criterion'] = random.choice(['gini', 'entropy'])
        
        if random.random() < indpb:
            individual['max_features'] = random.choice([None, 'sqrt', 'log2'])
        
        if random.random() < indpb:
            individual['bootstrap'] = random.choice([True, False])
        
        return (individual,)
    
    def _crossover(self, ind1, ind2):
        for key in ind1.keys():
            if random.random() < 0.5:
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2
    
    def setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._init_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evolve(self):
        self.setup_toolbox()
        pop = self.toolbox.population(n=50)
        result = algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=True)
        best_individual = tools.selBest(pop, 1)[0]
        print(f"Best individual: {best_individual} with fitness {best_individual.fitness.values}")
        return best_individual

if __name__ == "__main__":
    import os
    import logging
    import warnings
    # Disable warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    eamodel = evolveRandomForest(X_train, y_train, X_val, y_val)
    eamodel.setup_toolbox()
    pop = eamodel.toolbox.population(n=50)
    result = algorithms.eaSimple(pop, eamodel.toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=True)
    best_individual = tools.selBest(pop, 1)[0]
    print(f"Best individual: {best_individual} with fitness {best_individual.fitness.values}")

