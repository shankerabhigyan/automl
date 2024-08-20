from data_handler import DataCleaner, DataSplitter
from sbe import SBE
import pandas as pd
from sklearn.datasets import fetch_california_housing

print("Loading dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

cleaner = DataCleaner()
splitter = DataSplitter()
print("Cleaning the data...")
df = cleaner.clean(df)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
print("Splitting the data into train, test, and validation sets...")
X_train, X_test, X_val, y_train, y_test, y_val = splitter.kfold_split(X, y)
print("Performing feature selection with Sequential Backward Elimination...")

sbe = SBE(epochs=10)
selected_features = sbe.sequentialBackwardElimination(X_train.values, y_train.values)
print(f"Selected features: {X_train.columns[selected_features]}")