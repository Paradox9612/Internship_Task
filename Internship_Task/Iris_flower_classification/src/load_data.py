import seaborn as sns
import pandas as pd

def load_iris_dataset():
    iris = sns.load_dataset('iris')
    X = iris.drop('species', axis=1)
    y = iris['species']
    return X, y