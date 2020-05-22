# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:01:14 2020

@author: Piotr Waledzik
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20) if pd.isnull(dataset.iloc[i,j]) == False])
    
# Most frequent pairs
from collections import Counter
from itertools import combinations

pairs = Counter()
for transaction in transactions:
    if len(transactions) < 2:
        continue
    transaction.sort()
    for combination in combinations(transaction, 2):
        pairs[combination] += 1
        
# 50 Most common pairs
print(pairs.most_common(10))
