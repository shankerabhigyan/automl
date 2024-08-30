from data_handler import DataCleaner, DataSplitter
from sbe import SBE
import pandas as pd
from sklearn.datasets import fetch_california_housing
import torch
import numpy as np
from deap import base, creator, tools, algorithms


print("Loading dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# ## adding two dummy features
# df['dummy1'] = np.random.normal(0, 100, len(df))
# df['dummy2'] = np.random.normal(0, 1000, len(df))

# cleaner = DataCleaner()
# splitter = DataSplitter()
# print("Cleaning the data...")
# df = cleaner.clean(df)
# X = df.drop(columns=['MedHouseVal'])
# y = df['MedHouseVal']
# print("Splitting the data into train, test, and validation sets...")
# X_train, X_test, X_val, y_train, y_test, y_val = splitter.kfold_split(X, y)
# print("Performing feature selection with Sequential Backward Elimination...")

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# sbe = SBE(epochs=300, lr=0.001, device=device)
# selected_features = sbe.sequentialBackwardElimination(X_train.values, y_train.values)
# print(f"All features: {X_train.columns}")
# print(f"Selected features: {X_train.columns[selected_features]}")

################# Model evolution testing #################
from model_evolution import evolveRegressionNN, BaseNN

cleaner = DataCleaner()
splitter = DataSplitter()
print("Cleaning the data...")
df = cleaner.clean(df)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
print("Splitting the data into train, test, and validation sets...")
X_train, X_test, X_val, y_train, y_test, y_val = splitter.kfold_split(X, y)

# # select selected features
# X_train = X_train.iloc[:, selected_features]
# X_test = X_test.iloc[:, selected_features]
# X_val = X_val.iloc[:, selected_features]
print(f"Train Test Val shapes: {X_train.shape, X_test.shape, X_val.shape}")

print("Evolving the model...")
eamodel = evolveRegressionNN(X_train.values, y_train.values, X_val.values, y_val.values)
eamodel.setup_toolbox()
pop = eamodel.toolbox.population(n=30)
print("Training the model...")
result = algorithms.eaSimple(pop, eamodel.toolbox, cxpb=0.7, mutpb=0.4, ngen=40, verbose=True)
best_individual = tools.selBest(pop, 1)[0]
print(f"Best individual: {best_individual} with fitness {best_individual.fitness.values}")
