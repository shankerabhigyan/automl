from utils import DataCleaner, DataSplitter, tSNE, DataEater, EDA, LRScheduler, StepLR, Vectorizer
from src.regression import BaseNN, proxyNN, SBE, evolveRegressionNN
from src.classification import Node, DecisionTree, RandomForest, evolveRandomForest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

# Pipeline

class autoModel:
    def __init__(self, data, target_col, task=None, random_state=42):
        self.data = data
        self.target_col = target_col
        self.random_state = 42
        self.task = task
        self.model = None
        self.data_X = None
        self.data_y = None
        self.val_X = None
        self.val_y = None

    def _read_data(self):
        de = DataEater(self.data,strategy='mean',target_column=self.target_col)
        de.fetch_data()
        de.clean()
        self.data = de.data

    def _set_task(self):
        self.eda = EDA(self.data)
        self.task = self.eda.find_task_type()
        print(f"Task type detected : {self.task}")

    def _plot_corr(self):
        self.eda.plot_corr()

    def _split_data(self):
        ds = DataSplitter()
        self.data_y = self.data[self.target_col]
        self.data_X = self.data.drop(self.target_col, axis=1)
        self.xtrain, self.ytrain, self.xtest, self.ytest = ds.simple_split(self.data_X, self.data_y, test_size=0.2)
        del self.data
        del self.data_X
        del self.data_y

    def _vectorize(self):
        vec = Vectorizer(self.xtrain, self.ytrain)
        self.xtrain = vec.vectorize_features()
        self.ytrain = vec.vectorize_targets()
        vec2 = Vectorizer(self.xtest, self.ytest)
        self.xtest = vec2.vectorize_features()
        self.ytest = vec2.vectorize_targets()

    def fit_data(self):
        self._read_data()
        # self._plot_corr()
        self._split_data()
        self._vectorize()
        print(f"Feature shape : {self.xtrain.shape}, Target shape : {self.ytrain.shape}")
        if self.task == 'regression':
            self.model = evolveRegressionNN(self.xtrain, self.ytrain, self.xtest, self.ytest)
            bestindividual = self.model.evolve()
            with open('best_individual.txt', 'w') as f:
                f.write(str(bestindividual))
            logbook = self.model.logbook
            print(f"Best individual: {bestindividual} with fitness {bestindividual.fitness.values}")
            pickle.dump(logbook, open('logbook.pkl', 'wb'))
        elif self.task == 'classification':
            self.model = evolveRandomForest(self.xtrain, self.ytrain, self.xtest, self.ytest)
            bestindividual = self.model.evolve()
            with open('best_individual.txt', 'w') as f:
                f.write(str(bestindividual))
        else:
            raise ValueError(f"Unsupported task type {self.task}")

    
if __name__=='__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--target', type=str, help='Target column')
    parser.add_argument('--task', type=str, help='Task type')
    args = parser.parse_args()

    am = autoModel(args.data, args.target, args.task)
    am.fit_data()