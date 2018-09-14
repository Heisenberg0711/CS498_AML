import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def label_income(row):
    if row[14] == "<=50K":
        return 0
    if row[14] == ">50K":
        return 1

df = pd.read_csv('train.data', header = None)
df = df.dropna(axis=0,how='any')
df.drop(columns = [1,3,5,6,7,8,9,13])
